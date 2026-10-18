"""3-D instance proposal baseline inspired by Open3DIS/OVIR-3D.

Small-object candidates are seeded by ConvNeXt-L, refined by MobileSAM, projected
to world 3-D, greedily associated across views, then written back with the
original large > small > floor priority. GT is read only for scoring.
"""
import json,gc,argparse
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from mobile_sam import sam_model_registry, SamPredictor
from scene import ROOT,dump
from score_voxel_map import depth_points
from score_saved import grid_indices
from analyze_small_objects import dense_labels,depth_weights

ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();BASE=args.base;OUT=args.out;OUT.mkdir(parents=True,exist_ok=True);N=600

def score(gt,pred,gh,hp,valid):
 cm=np.zeros((4,4),int);np.add.at(cm,(gt[valid],pred[valid]),1); rows=[]
 for k,n in [(1,'floor'),(2,'small'),(3,'large')]:
  g=cm[k].sum();p=cm[:,k].sum();rows.append({'class':n,'gt':int(g),'correct':int(cm[k,k]),'predicted':int(p),'recall':100*cm[k,k]/g if g else None,'precision':100*cm[k,k]/p if p else None})
 m=valid&np.isfinite(hp)
 return {'accuracy':100*cm.trace()/valid.sum(),'correct':int(cm.trace()),'GT':int(valid.sum()),'classes':rows,'height_MAE_cm':None if not m.any() else 100*float(np.abs(hp[m]-gh[m]).mean()),'cm':cm.tolist()}

def run(room):
 p=BASE/f'room-{room:02d}';d=json.loads((p/'design.json').read_text());poses=d['poses'];
 if (p/'fastlio-poses.json').exists():
  from fastlio_pose import load_fastlio_poses; poses=load_fastlio_poses(p/'fastlio-poses.json',d)
 state=torch.load(p/'augmented/saved-map.pt',map_location='cpu',weights_only=True);feats=torch.load(ROOT/f'reproduction/map_segmentation/multi_room/small_object_analysis/room-{room:02d}/cached-dense-features.pt',map_location='cpu',weights_only=True)
 lut=np.full(44,3,np.uint8);lut[[1,15]]=1;lut[40:43]=2;lut[16]=4;lut[43]=0
 predictor=SamPredictor(sam_model_registry['vit_t'](checkpoint=str(ROOT/'weights/mobile_sam.pt')).cuda().eval())
 cnt=np.zeros((8,N,N,4),np.uint16); covered=np.zeros((N,N),bool); hmax=np.full((N,N),-np.inf,np.float32); components=[]
 for pose,feat in zip(poses,feats):
  v=pose['index']; dep=np.load(p/f'augmented/view-{v:02d}-geometry.npz')['depth'];good=(dep>0)&np.isfinite(dep)
  arch=np.asarray(Image.open(ROOT/f'reproduction/map_segmentation/multi_room/small_object_analysis/room-{room:02d}'/f'view-{v:02d}-frontend-classes.png')); fine=arch.copy()
  rgb=np.asarray(Image.open(p/f'augmented/view-{v:02d}-rgb.png'))[:,:,:3];predictor.set_image(rgb)
  seed=((fine>=40)&(fine<=42)).astype(np.uint8);num,comp,stats,_=cv2.connectedComponentsWithStats(seed,8)
  for ci in range(1,num):
   x,y,w,h,a=stats[ci]
   if a<3:continue
   box=np.array([max(0,x-2),max(0,y-2),min(639,x+w+2),min(639,y+h+2)],np.float32);masks,_,_=predictor.predict(box=box[None],multimask_output=False);m=masks[0]
   if (m&(comp==ci)).sum()<2:continue
   region=m&(cv2.dilate((comp==ci).astype(np.uint8),np.ones((9,9),np.uint8))>0);fine[region]=int(np.bincount(fine[comp==ci],minlength=44).argmax())
  pts=depth_points(dep,pose);world=pts[good];lab=lut[fine[good]];wgt=depth_weights(dep,state['mapping_config'])[good];i,j=grid_indices(world[:,0],world[:,2],state);inside=(i>=0)&(i<N)&(j>=0)&(j<N)&(lab>=1)&(lab<=3)&(wgt>1e-5);i=i[inside];j=j[inside];lab=lab[inside];world=world[inside];np.add.at(cnt[v],(i,j,lab),1);np.maximum.at(hmax,(i,j),world[:,1]-d['floor_y']);covered[i,j]=True
  # 3-D proposal components from refined small mask.
  small=(fine>=40)&(fine<=42);num,cc,st,_=cv2.connectedComponentsWithStats(small.astype(np.uint8),8)
  for ci in range(1,num):
   yy,xx=np.where(cc==ci)
   if len(xx)<3:continue
   pix=np.zeros_like(good);pix[yy,xx]=True;ok=pix&good;pw=pts[ok];
   if len(pw)<3:continue
   components.append({'view':v,'centroid':pw.mean(0),'points':pw,'label':int(np.bincount(fine[ok],minlength=44).argmax()),'xy':set(zip(i[(pix[good])[inside]],j[(pix[good])[inside]])) if False else None})
 # Existing view-balanced large/floor evidence; instances only replace small labels.
 total=cnt[:,:,:,1:4].sum(-1);frac=np.divide(cnt[:,:,:,1:4],total[:,:,:,None],out=np.zeros_like(cnt[:,:,:,1:4],np.float32),where=total[:,:,:,None]>0);support=(frac>=np.array([.45,.45,.60])[None,None,None,:]).sum(0);cand=np.zeros((N,N,3),bool);cand[:,:,0]=covered;cand[:,:,1:]=support[:,:,1:]>=1;pred=np.max(cand*np.array([1,2,3],np.uint8),-1).astype(np.uint8);hp=hmax.copy();hp[pred==0]=np.nan
 # Associate proposals by 3-D centroid, collect labels and rasterize their projected points.
 clusters=[]
 for q in components:
  hit=None
  for c in clusters:
   if np.linalg.norm(q['centroid']-c['centroid'])<.18:hit=c;break
  if hit is None:hit={'centroid':q['centroid'].copy(),'items':[]};clusters.append(hit)
  hit['items'].append(q);hit['centroid']=np.mean([z['centroid'] for z in hit['items']],0)
 for c in clusters:
  labels=[z['label'] for z in c['items']]; coarse=2 if sum(40<=x<=42 for x in labels)>=len(labels)/2 else 3
  for z in c['items']:
   q=z['points'];ii,jj=grid_indices(q[:,0],q[:,2],state);inside=(ii>=0)&(ii<N)&(jj>=0)&(jj<N);ii=ii[inside];jj=jj[inside]
   # large evidence retains priority; otherwise the 3-D instance writes small.
   # Require cross-view support for small instances.  A single SAM proposal
   # is kept as evidence but cannot overwrite the fused semantic grid.
   if coarse==2 and len(c['items'])>=2: pred[ii,jj]=np.maximum(pred[ii,jj],2)
   else: pred[ii,jj]=3
 old=np.load(p/'semantic_elevation/maps.npz');rep=score(old['GT_semantic'],pred,old['GT_height'],hp,old['valid_GT']);rep.update({'room':room,'instances':len(clusters),'proposals':len(components),'sam':True,'priority':'large > small > floor'});dump(OUT/f'room-{room:02d}.json',rep);np.savez_compressed(OUT/f'room-{room:02d}.npz',GT=old['GT_semantic'],valid=old['valid_GT'],pred=pred,GT_height=old['GT_height'],pred_height=hp);return rep

reports=[run(r) for r in [10,9,17]];gt=sum(x['GT'] for x in reports);summary={'rooms':reports,'pooled_accuracy':100*sum(x['correct'] for x in reports)/gt,'GT':gt,'correct':sum(x['correct'] for x in reports)};dump(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))
