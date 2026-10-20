"""3-D instance proposal baseline inspired by Open3DIS/OVIR-3D.

Small-object candidates are seeded by ConvNeXt-L, refined by MobileSAM, projected
to world 3-D, greedily associated across views, then written back with the
original large > small > floor priority. GT is read only for scoring.
"""
import json,gc,argparse
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from mobile_sam import sam_model_registry, SamPredictor
from scene import ROOT,dump
from score_voxel_map import depth_points
from score_saved import grid_indices
from analyze_small_objects import dense_labels,depth_weights
from small_object_fusion import bayesian_update, categorical_likelihood, geometric_small_candidate_mask

ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--rooms',type=int,nargs='+',default=[10,9,17],help='room ids to evaluate');ap.add_argument('--consistency-threshold',type=float,default=0.0,help='drop views whose non-GT semantic agreement with other views is below this value');ap.add_argument('--drop-lowest',type=int,default=0,help='drop this many lowest-consistency views (relative, no GT)');ap.add_argument('--adaptive-consistency',action='store_true',help='gate only when the view-consistency distribution has a reliable high-consistency cluster');ap.add_argument('--no-sam',action='store_true',help='ablation: use ConvNeXt-L frontend labels without MobileSAM refinement');ap.add_argument('--small-support-views',type=int,default=2,help='minimum number of supporting views for a small 3-D instance');ap.add_argument('--small-object-recovery',action='store_true',help='add geometry-seeded high-resolution crops before SAM; recovers small objects missed by full-frame ConvNeXt-L');ap.add_argument('--small-recovery-threshold',type=float,default=.20,help='minimum crop small-object pixel fraction for a geometric proposal');ap.add_argument('--small-recovery-min-height',type=float,default=.02,help='minimum candidate height above a fitted floor plane in metres');ap.add_argument('--small-recovery-max-height',type=float,default=.45,help='maximum candidate height above a fitted floor plane in metres');ap.add_argument('--small-recovery-min-area',type=int,default=3);ap.add_argument('--small-recovery-max-area',type=int,default=2500);ap.add_argument('--small-recovery-max-candidates',type=int,default=12);ap.add_argument('--small-recovery-crop-margin',type=int,default=16);ap.add_argument('--small-posterior-threshold',type=float,default=.55,help='minimum fused small-object posterior for a 3-D proposal');ap.add_argument('--no-instance-fusion',action='store_true',help='ablation: disable 3-D instance proposals after semantic projection');ap.add_argument('--semantic-source',choices=['frontend','gt'],default='frontend',help='diagnostic only: use cached ConvNeXt-L labels or per-view semantic GT labels');ap.add_argument('--min-floor-views',type=int,default=1);ap.add_argument('--min-small-views',type=int,default=1);ap.add_argument('--min-large-views',type=int,default=1);ap.add_argument('--floor-fraction',type=float,default=.45);ap.add_argument('--small-fraction',type=float,default=.45);ap.add_argument('--large-fraction',type=float,default=.60);ap.add_argument('--large-fraction-auto',action='store_true',help='use the tuned large threshold only when the consistency gate actually removes views');ap.add_argument('--floor-height-threshold',type=float,default=0.0,help='optional geometric floor prior in metres above the room datum');ap.add_argument('--floor-prior-classes',choices=['all','large'],default='all',help='classes affected by the optional floor prior');ap.add_argument('--floor-prior-preserve-small',action='store_true',help='never overwrite small-object evidence with the geometric floor prior');ap.add_argument('--floor-prior-auto',action='store_true',help='enable floor prior only when the projected frontend has enough floor evidence');ap.add_argument('--floor-prior-min-ratio',type=float,default=.42);ap.add_argument('--online-ground-plane',action='store_true',help='estimate a ground plane from projected FAST-LIO point clouds and floor candidates');ap.add_argument('--multi-ground-planes',action='store_true',help='fit multiple horizontal planes for stair treads');ap.add_argument('--ground-plane-max-planes',type=int,default=6);ap.add_argument('--ground-plane-secondary-min-inliers',type=int,default=50);ap.add_argument('--ground-plane-inlier-threshold',type=float,default=.03);ap.add_argument('--ground-plane-min-inliers',type=int,default=200);ap.add_argument('--robust-fusion',action='store_true',help='recommended deployable fusion: disable instance overwrite, gate views, and enable conditional geometry priors');args=ap.parse_args();
if args.robust_fusion:
 args.no_instance_fusion=True;args.adaptive_consistency=True;args.large_fraction=.9;args.large_fraction_auto=True;args.floor_height_threshold=.05;args.floor_prior_auto=True;args.online_ground_plane=True;args.multi_ground_planes=True
BASE=args.base;OUT=args.out;OUT.mkdir(parents=True,exist_ok=True);N=600

def score(gt,pred,gh,hp,valid):
 cm=np.zeros((4,4),int);np.add.at(cm,(gt[valid],pred[valid]),1); rows=[]
 for k,n in [(1,'floor'),(2,'small'),(3,'large')]:
  g=cm[k].sum();p=cm[:,k].sum();rows.append({'class':n,'gt':int(g),'correct':int(cm[k,k]),'predicted':int(p),'recall':100*cm[k,k]/g if g else None,'precision':100*cm[k,k]/p if p else None})
 m=valid&np.isfinite(hp)
 return {'accuracy':100*cm.trace()/valid.sum(),'correct':int(cm.trace()),'GT':int(valid.sum()),'classes':rows,'height_MAE_cm':None if not m.any() else 100*float(np.abs(hp[m]-gh[m]).mean()),'cm':cm.tolist()}

def estimate_ground_plane(room_dir, room, poses, lut, semantic_source, inlier_threshold, min_inliers):
 """Estimate an up-facing ground plane from FAST-LIO-projected floor candidates.

 The estimator never reads ``floor_y`` or GT geometry.  It uses only depth,
 estimated poses, and the per-view frontend floor labels.  RANSAC rejects
 furniture/wall outliers, then refits the plane by least squares.
 """
 candidates=[]
 for pose in poses:
  v=int(pose['index']); dep=np.load(room_dir/f'augmented/view-{v:02d}-geometry.npz')['depth']; good=(dep>0)&np.isfinite(dep)
  if semantic_source=='gt':
   fine=np.asarray(Image.open(room_dir/f'augmented/view-{v:02d}-gt.png')).copy(); fine[fine>=44]=43
  else:
   fine=np.asarray(Image.open(ROOT/f'reproduction/map_segmentation/multi_room/small_object_analysis/room-{room:02d}'/f'view-{v:02d}-frontend-classes.png'))
  floor=good&(lut[fine]==1); pts=depth_points(dep,pose)[floor]
  if len(pts)>0:
   if len(pts)>4000: pts=pts[np.linspace(0,len(pts)-1,4000).astype(int)]
   candidates.append(pts)
 if not candidates: return None
 pts=np.concatenate(candidates,axis=0)
 if len(pts)<min_inliers: return None
 rng=np.random.default_rng(1000+room); best=None; threshold=float(inlier_threshold)
 for _ in range(800):
  tri=pts[rng.choice(len(pts),3,replace=False)]; n=np.cross(tri[1]-tri[0],tri[2]-tri[0]); norm=np.linalg.norm(n)
  if norm<1e-8: continue
  n=n/norm
  if n[1]<0: n=-n
  if n[1]<0.94: continue
  d0=-float(n@tri[0]); dist=np.abs(pts@n+d0); inliers=dist<=threshold; count=int(inliers.sum())
  med=float(np.median(dist[inliers])) if count else 1e9
  if best is None or count>best[0] or (count==best[0] and med<best[1]): best=(count,med,inliers,n,d0)
 if best is None or best[0]<min_inliers: return None
 inliers=best[2]; centered=pts[inliers]-pts[inliers].mean(0); _,_,vh=np.linalg.svd(centered,full_matrices=False); n=vh[-1]; n=n/np.linalg.norm(n)
 if n[1]<0: n=-n
 d0=-float(n@pts[inliers].mean(0)); residual=np.abs(pts@n+d0)
 return {'normal':n.tolist(),'offset':d0,'candidate_points':int(len(pts)),'inlier_points':int(inliers.sum()),'inlier_ratio':float(inliers.mean()),'median_abs_residual_m':float(np.median(residual[inliers])),'p95_abs_residual_m':float(np.percentile(residual[inliers],95))}

def estimate_ground_planes(room_dir, room, poses, lut, semantic_source, inlier_threshold, min_inliers, max_planes=6, secondary_min_inliers=50):
 """Fit several horizontal planes, allowing stair treads at different heights."""
 candidates=[]
 for pose in poses:
  v=int(pose['index']); dep=np.load(room_dir/f'augmented/view-{v:02d}-geometry.npz')['depth']; good=(dep>0)&np.isfinite(dep)
  if semantic_source=='gt': fine=np.asarray(Image.open(room_dir/f'augmented/view-{v:02d}-gt.png')).copy(); fine[fine>=44]=43
  else: fine=np.asarray(Image.open(ROOT/f'reproduction/map_segmentation/multi_room/small_object_analysis/room-{room:02d}'/f'view-{v:02d}-frontend-classes.png'))
  pts=depth_points(dep,pose)[good&(lut[fine]==1)]
  if len(pts): candidates.append(pts[np.linspace(0,len(pts)-1,min(len(pts),4000)).astype(int)])
 if not candidates: return []
 remaining=np.concatenate(candidates,axis=0); all_count=len(remaining); planes=[]; rng=np.random.default_rng(2000+room)
 for plane_id in range(max_planes):
  need=min_inliers if plane_id==0 else secondary_min_inliers
  if len(remaining)<need: break
  best=None
  for _ in range(800):
   tri=remaining[rng.choice(len(remaining),3,replace=False)]; n=np.cross(tri[1]-tri[0],tri[2]-tri[0]); norm=np.linalg.norm(n)
   if norm<1e-8: continue
   n=n/norm
   if n[1]<0: n=-n
   if n[1]<0.98: continue
   d0=-float(n@tri[0]); dist=np.abs(remaining@n+d0); mask=dist<=inlier_threshold; count=int(mask.sum())
   if best is None or count>best[0]: best=(count,mask,n,d0)
  if best is None or best[0]<need: break
  mask=best[1]; centered=remaining[mask]-remaining[mask].mean(0);_,_,vh=np.linalg.svd(centered,full_matrices=False);n=vh[-1];n=n/np.linalg.norm(n)
  if n[1]<0:n=-n
  d0=-float(n@remaining[mask].mean(0)); residual=np.abs(remaining@n+d0)
  planes.append({'plane_id':plane_id,'normal':n.tolist(),'offset':d0,'candidate_points':all_count,'inlier_points':int(mask.sum()),'inlier_ratio':float(mask.mean()),'median_abs_residual_m':float(np.median(residual[mask])),'p95_abs_residual_m':float(np.percentile(residual[mask],95)),'height_reference_m':float(-d0/n[1])})
  remaining=remaining[~mask]
 return planes


def _plane_height_candidates(world, valid, planes, floor_y):
 if planes:
  distances=np.stack([world@np.asarray(q['normal'])+float(q['offset']) for q in planes],axis=-1)
  # A point is a proposal when it is just above at least one fitted support
  # plane.  This catches objects on stair treads without treating the plane
  # itself as an obstacle.
  return np.any((distances>=args.small_recovery_min_height)&(distances<=args.small_recovery_max_height),axis=-1)&valid
 return geometric_small_candidate_mask(world[...,1],valid,floor_y,args.small_recovery_min_height,args.small_recovery_max_height)


def recover_small_candidates(rgb, world, valid, fine, text, model, planes, floor_y):
 """Seed missed small objects with geometric proposals and high-res CLIP crops."""
 if model is None:
  return fine, np.zeros((*valid.shape,3),np.float32), 0
 candidates=_plane_height_candidates(world,valid,planes,floor_y)
 num,components,stats,_=cv2.connectedComponentsWithStats(candidates.astype(np.uint8),8)
 ranked=[]
 for ci in range(1,num):
  x,y,w,h,area=stats[ci]
  if area<args.small_recovery_min_area or area>args.small_recovery_max_area: continue
  ranked.append((int(area),ci,int(x),int(y),int(w),int(h)))
 ranked=sorted(ranked,reverse=True)[:args.small_recovery_max_candidates]
 probs_map=np.zeros((*valid.shape,3),np.float32);accepted=0
 for _,ci,x,y,w,h in ranked:
  x0=max(0,x-args.small_recovery_crop_margin);y0=max(0,y-args.small_recovery_crop_margin)
  x1=min(rgb.shape[1],x+w+args.small_recovery_crop_margin);y1=min(rgb.shape[0],y+h+args.small_recovery_crop_margin)
  crop=rgb[y0:y1,x0:x1]
  with torch.inference_mode():
   feat=model.get_image_features(crop.transpose(2,0,1))[0]
   scores=torch.einsum('chw,kc->khw',feat,text.to(feat.device))
   scores=F.interpolate(scores[None],size=(y1-y0,x1-x0),mode='bilinear',align_corners=False)[0]
  region=(components[y0:y1,x0:x1]==ci)
  if not region.any(): continue
  labels=scores.argmax(0).detach().cpu().numpy().astype(np.uint8)
  ids=labels[region];small_ids=ids[np.isin(ids,[40,41,42])]
  small_fraction=len(small_ids)/len(ids)
  if small_fraction<args.small_recovery_threshold: continue
  probs=categorical_likelihood(ids)
  small_label=int(np.bincount(small_ids,minlength=44)[40:43].argmax()+40)
  target=(components==ci)
  fine[target]=small_label
  probs_map[target]=probs.astype(np.float32)
  accepted+=1
 return fine,probs_map,accepted


def label_posterior(label):
 if 40<=int(label)<=42: return np.array([.05,.90,.05],np.float64)
 if int(label)==1 or int(label)==15: return np.array([.90,.05,.05],np.float64)
 return np.array([.05,.05,.90],np.float64)

def run(room):
 p=BASE/f'room-{room:02d}';d=json.loads((p/'design.json').read_text());poses=d['poses'];
 if (p/'fastlio-poses.json').exists():
  from fastlio_pose import load_fastlio_poses; poses=load_fastlio_poses(p/'fastlio-poses.json',d)
 state=torch.load(p/'augmented/saved-map.pt',map_location='cpu',weights_only=True);feats=torch.load(ROOT/f'reproduction/map_segmentation/multi_room/small_object_analysis/room-{room:02d}/cached-dense-features.pt',map_location='cpu',weights_only=True)
 lut=np.full(44,3,np.uint8);lut[[1,15]]=1;lut[40:43]=2;lut[16]=4;lut[43]=0
 floor_ratio_samples=[]
 if args.floor_prior_auto and args.floor_height_threshold>0:
  for q in poses:
   vv=int(q['index'])
   if args.semantic_source=='gt': ff=np.asarray(Image.open(p/f'augmented/view-{vv:02d}-gt.png')).copy();ff[ff>=44]=43
   else: ff=np.asarray(Image.open(ROOT/f'reproduction/map_segmentation/multi_room/small_object_analysis/room-{room:02d}'/f'view-{vv:02d}-frontend-classes.png'))
   floor_ratio_samples.append(float(np.mean(lut[ff]==1)))
 floor_prior_enabled=args.floor_height_threshold>0 and (not args.floor_prior_auto or float(np.mean(floor_ratio_samples))>=args.floor_prior_min_ratio)
 ground_planes=estimate_ground_planes(p,room,poses,lut,args.semantic_source,args.ground_plane_inlier_threshold,args.ground_plane_min_inliers,args.ground_plane_max_planes,args.ground_plane_secondary_min_inliers) if args.online_ground_plane and args.multi_ground_planes else ([] if args.online_ground_plane else None)
 if args.online_ground_plane and not args.multi_ground_planes:
  primary=estimate_ground_plane(p,room,poses,lut,args.semantic_source,args.ground_plane_inlier_threshold,args.ground_plane_min_inliers);ground_planes=[] if primary is None else [primary]
 ground_planes=[q for q in (ground_planes or []) if q['normal'][1]>=.98 and q['inlier_ratio']>=.05]
 ground_plane=ground_planes[0] if ground_planes else None
 if args.online_ground_plane: floor_prior_enabled=ground_plane is not None and (not args.floor_prior_auto or float(np.mean(floor_ratio_samples))>=args.floor_prior_min_ratio)
 predictor=None if args.no_sam else SamPredictor(sam_model_registry['vit_t'](checkpoint=str(ROOT/'weights/mobile_sam.pt')).cuda().eval())
 recovery_model=None
 if args.small_object_recovery and args.semantic_source=='frontend':
  from vision_models.clip_dense import ClipModel
  recovery_model=ClipModel(str(ROOT/'weights/clip.pth'),jetson=False).eval()
 cnt=np.zeros((8,N,N,4),np.uint16); covered=np.zeros((N,N),bool); hmax_views=np.full((8,N,N),-np.inf,np.float32); components=[];recovery_counts=[]
 for pose in poses:
  # The cached frontend features are indexed by the original view id.  Using
  # the pose index here also lets robustness tests evaluate shuffled/subsampled
  # view sets without silently pairing a pose with the wrong image features.
  feat=feats[int(pose['index'])]
  v=pose['index']; dep=np.load(p/f'augmented/view-{v:02d}-geometry.npz')['depth'];good=(dep>0)&np.isfinite(dep)
  if args.semantic_source=='gt': fine=np.asarray(Image.open(p/f'augmented/view-{v:02d}-gt.png')).copy(); fine[fine>=44]=43
  else: arch=np.asarray(Image.open(ROOT/f'reproduction/map_segmentation/multi_room/small_object_analysis/room-{room:02d}'/f'view-{v:02d}-frontend-classes.png')); fine=arch.copy()
  rgb=np.asarray(Image.open(p/f'augmented/view-{v:02d}-rgb.png'))[:,:,:3]
  recovery_prob_map=np.zeros((*fine.shape,3),np.float32);recovered=0
  if recovery_model is not None:
   recovery_prob_map_input=depth_points(dep,pose)
   fine,recovery_prob_map,recovered=recover_small_candidates(rgb,recovery_prob_map_input,good,fine,state['text_features'],recovery_model,ground_planes,d['floor_y'])
  recovery_counts.append({'view':int(v),'accepted_candidates':int(recovered)})
  if predictor is not None:
   predictor.set_image(rgb)
   seed=((fine>=40)&(fine<=42)).astype(np.uint8);num,comp,stats,_=cv2.connectedComponentsWithStats(seed,8)
   for ci in range(1,num):
    x,y,w,h,a=stats[ci]
    if a<3:continue
    box=np.array([max(0,x-2),max(0,y-2),min(639,x+w+2),min(639,y+h+2)],np.float32);masks,_,_=predictor.predict(box=box[None],multimask_output=False);m=masks[0]
    if (m&(comp==ci)).sum()<2:continue
    region=m&(cv2.dilate((comp==ci).astype(np.uint8),np.ones((9,9),np.uint8))>0);fine[region]=int(np.bincount(fine[comp==ci],minlength=44).argmax())
  pts=depth_points(dep,pose);world=pts[good];lab=lut[fine[good]]
  if floor_prior_enabled:
   if ground_planes:
    distances=np.stack([world@np.asarray(q['normal'])+float(q['offset']) for q in ground_planes],axis=1); floor_mask=np.any((distances>=-0.03)&(distances<=args.floor_height_threshold),axis=1)
   else: floor_mask=world[:,1]-d['floor_y']<=args.floor_height_threshold
   if args.floor_prior_classes=='large': floor_mask &= lab==3
   if args.floor_prior_preserve_small: floor_mask &= lab!=2
   lab[floor_mask]=1
  wgt=depth_weights(dep,state['mapping_config'])[good];i,j=grid_indices(world[:,0],world[:,2],state);inside=(i>=0)&(i<N)&(j>=0)&(j<N)&(lab>=1)&(lab<=3)&(wgt>1e-5);i=i[inside];j=j[inside];lab=lab[inside];world=world[inside];np.add.at(cnt[v],(i,j,lab),1);np.maximum.at(hmax_views[v],(i,j),world[:,1]-d['floor_y']);covered[i,j]=True
  # 3-D proposal components from refined small mask.
  if args.no_instance_fusion: continue
  small=(fine>=40)&(fine<=42);num,cc,st,_=cv2.connectedComponentsWithStats(small.astype(np.uint8),8)
  for ci in range(1,num):
   yy,xx=np.where(cc==ci)
   if len(xx)<3:continue
   pix=np.zeros_like(good);pix[yy,xx]=True;ok=pix&good;pw=pts[ok];
   if len(pw)<3:continue
   label=int(np.bincount(fine[ok],minlength=44).argmax())
   evidence=recovery_prob_map[ok].mean(0)
   probs=evidence if evidence.sum()>0 else label_posterior(label)
   components.append({'view':v,'centroid':pw.mean(0),'points':pw,'label':label,'probs':probs.tolist(),'xy':set(zip(i[(pix[good])[inside]],j[(pix[good])[inside]])) if False else None})
 # Existing view-balanced large/floor evidence; instances only replace small labels.
 active=[int(p['index']) for p in poses];total_all=cnt.sum(0);view_consistency={};keep=[]
 for v in active:
  own=cnt[v].argmax(-1);other=(total_all-cnt[v]).argmax(-1);over=(cnt[v].sum(-1)>0)&((total_all-cnt[v]).sum(-1)>0);agree=over&(own==other);view_consistency[str(v)]=float(agree.sum()/over.sum()) if over.any() else 0.0
  if args.consistency_threshold<=0 or view_consistency[str(v)]>=args.consistency_threshold: keep.append(v)
 if args.drop_lowest>0 and len(active)>1:
  ranked=sorted(active,key=lambda v:view_consistency[str(v)])
  keep=[v for v in active if v not in set(ranked[:min(args.drop_lowest,len(active)-1)])]
 if args.adaptive_consistency and len(active)>=4:
  vals=np.asarray([view_consistency[str(v)] for v in active]);ranked=sorted(active,key=lambda v:view_consistency[str(v)]);median=float(np.median(vals));gap=median-view_consistency[str(ranked[1])]
  # A low median indicates a globally inconsistent pose set; in that case
  # dropping views only reduces coverage. Gate only a clear lower tail.
  if median>=0.62 and gap>=0.05:
   keep=[v for v in active if v not in set(ranked[:2])]
 if not keep: keep=active
 large_fraction_used=args.large_fraction
 if args.large_fraction_auto and len(keep)==len(active): large_fraction_used=.60
 fused_cnt=cnt[keep].sum(0);view_cnt=cnt[keep][:,:,:,1:4];view_total=view_cnt.sum(-1);view_frac=np.divide(view_cnt,view_total[:,:,:,None],out=np.zeros_like(view_cnt,np.float32),where=view_total[:,:,:,None]>0);support=(view_frac>=np.array([args.floor_fraction,args.small_fraction,large_fraction_used])[None,None,None,:]).sum(0);cand=np.zeros((N,N,3),bool);cand[:,:,0]=fused_cnt[:,:,1:4].sum(-1)>0;cand[:,:,1:]=support[:,:,1:]>=np.array([args.min_small_views,args.min_large_views])[None,None,:];pred=np.max(cand*np.array([1,2,3],np.uint8),-1).astype(np.uint8);hp=np.max(hmax_views[keep],axis=0);hp[pred==0]=np.nan
 # Associate proposals by 3-D centroid, collect labels and rasterize their projected points.
 clusters=[]
 for q in components:
  hit=None
  for c in clusters:
   if np.linalg.norm(q['centroid']-c['centroid'])<.18:hit=c;break
  if hit is None:hit={'centroid':q['centroid'].copy(),'items':[]};clusters.append(hit)
  hit['items'].append(q);hit['centroid']=np.mean([z['centroid'] for z in hit['items']],0)
 for c in clusters:
  posterior=np.full(3,1/3,dtype=np.float64)
  for z in c['items']: posterior=bayesian_update(posterior,z['probs'])
  c['posterior']=posterior.tolist(); coarse=2 if posterior[1]>=args.small_posterior_threshold else 3
  for z in c['items']:
   if z['view'] not in keep: continue
   q=z['points'];ii,jj=grid_indices(q[:,0],q[:,2],state);inside=(ii>=0)&(ii<N)&(jj>=0)&(jj<N);ii=ii[inside];jj=jj[inside]
   # Large evidence already present in ``pred`` retains priority.  A proposal
   # that does not pass the small-object posterior/support gate is only
   # evidence; it must not be promoted to a large object by default.
   if coarse==2 and len(c['items'])>=args.small_support_views:
    pred[ii,jj]=np.maximum(pred[ii,jj],2)
 old=np.load(p/'semantic_elevation/maps.npz');valid_gt=old['valid_GT'];rep=score(old['GT_semantic'],pred,old['GT_height'],hp,valid_gt);covered_valid=(pred>0)&valid_gt;observed=score(old['GT_semantic'],pred,old['GT_height'],hp,covered_valid) if covered_valid.any() else None;rep.update({'room':room,'instances':len(clusters),'proposals':len(components),'sam':not args.no_sam,'semantic_source':args.semantic_source,'priority':'large > small > floor','small_support_views':args.small_support_views,'small_object_recovery':args.small_object_recovery,'small_recovery_counts':recovery_counts,'small_recovery_threshold':args.small_recovery_threshold,'small_posterior_threshold':args.small_posterior_threshold,'min_support_views':[args.min_floor_views,args.min_small_views,args.min_large_views],'large_fraction_used':large_fraction_used,'instance_fusion':not args.no_instance_fusion,'floor_prior_enabled':floor_prior_enabled,'floor_prior_source':'online_plane' if ground_plane is not None else ('fixed_datum' if floor_prior_enabled else 'disabled'),'ground_plane':ground_plane,'ground_planes':ground_planes,'floor_ratio_mean':None if not floor_ratio_samples else float(np.mean(floor_ratio_samples)),'coverage_percent':100*float(covered_valid.sum())/float(valid_gt.sum()) if valid_gt.any() else 0.0,'observed_accuracy':None if observed is None else observed['accuracy'],'observed_GT':0 if observed is None else observed['GT'],'view_consistency':view_consistency,'kept_views':keep,'consistency_threshold':args.consistency_threshold});dump(OUT/f'room-{room:02d}.json',rep);np.savez_compressed(OUT/f'room-{room:02d}.npz',GT=old['GT_semantic'],valid=valid_gt,pred=pred,GT_height=old['GT_height'],pred_height=hp,covered=covered_valid);return rep

reports=[run(r) for r in args.rooms];gt=sum(x['GT'] for x in reports);summary={'rooms':reports,'pooled_accuracy':100*sum(x['correct'] for x in reports)/gt,'GT':gt,'correct':sum(x['correct'] for x in reports)};dump(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))
