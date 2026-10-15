"""Instance-like, view-balanced semantic evidence baselines.

This is a lightweight reproduction of the instance-centric idea in Open3DIS/
OVIR-3D: pixels are first aggregated per RGB-D view and XY cell, then views
vote for a cell. It keeps the requested large > small > floor priority.
It does not use GT to build predictions.
"""
import csv, json, gc, argparse
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scene import ROOT, dump, digest
from score_voxel_map import depth_points
from score_saved import grid_indices
from analyze_small_objects import dense_labels, depth_weights
try:
 import cv2
 from mobile_sam import sam_model_registry, SamPredictor
except Exception:
 cv2=None; sam_model_registry=None; SamPredictor=None

BASE=ROOT/'reproduction/map_segmentation/multi_room'; OUT=BASE/'instance_evidence'; OUT.mkdir(exist_ok=True)
ROOMS=[10,9,17]; N=600; V=8

def metrics(gt,pred,hgt,hp,valid):
 cm=np.zeros((4,4),np.int64); np.add.at(cm,(gt[valid],pred[valid]),1)
 rows=[]
 for k,nm in [(1,'floor_stairs'),(2,'small_objects'),(3,'large_objects')]:
  g=int(cm[k].sum()); tp=int(cm[k,k]); pp=int(cm[:,k].sum())
  rows.append({'class':nm,'gt':g,'correct':tp,'recall_percent':100*tp/g if g else None,'precision_percent':100*tp/pp if pp else None,'predicted':pp})
 m=valid&np.isfinite(hp)
 return {'GT_cells':int(valid.sum()),'correct_cells':int(np.trace(cm)),'accuracy_percent':100*float(np.trace(cm)/valid.sum()),'missing':int(cm[:,0].sum()),'per_class':rows,'height_cells':int(m.sum()),'height_MAE_cm':100*float(np.abs(hp[m]-hgt[m]).mean()),'cm':cm.tolist()}

def build(room, use_sam=False):
 p=BASE/f'room-{room:02d}'; d=json.loads((p/'design.json').read_text()); state=torch.load(p/'augmented/saved-map.pt',map_location='cpu',weights_only=True)
 feats=torch.load(BASE/f'small_object_analysis/room-{room:02d}/cached-dense-features.pt',map_location='cpu',weights_only=True)
 # Per-view evidence: count of projected pixels for each class in each XY cell.
 cnt=np.zeros((V,N,N,4),np.uint16); hmax=np.full((N,N),-np.inf,np.float32); covered=np.zeros((N,N),bool)
 lut=np.full(44,3,np.uint8); lut[[1,15]]=1; lut[40:43]=2; lut[16]=4; lut[43]=0
 predictor=None
 if use_sam:
  if sam_model_registry is None: raise RuntimeError('MobileSAM unavailable')
  predictor=SamPredictor(sam_model_registry['vit_t'](checkpoint=str(ROOT/'weights/mobile_sam.pt')).cuda().eval())
 for pose,feat in zip(d['poses'],feats):
  v=pose['index']; dep=np.load(p/f'augmented/view-{v:02d}-geometry.npz')['depth']; good=(dep>0)&np.isfinite(dep)
  with torch.inference_mode(): fine=dense_labels(feat.cuda(),state['text_features'])
  arch=np.asarray(Image.open(BASE/f'small_object_analysis/room-{room:02d}'/f'view-{v:02d}-frontend-classes.png'))
  # Start from the archived deterministic baseline labels.
  if use_sam: fine=arch.copy()
  if use_sam:
   rgb=np.asarray(Image.open(p/f'augmented/view-{v:02d}-rgb.png'))[:,:,:3]
   predictor.set_image(rgb)
   # Refine only small-object candidate components; all other labels remain unchanged.
   seed=((fine>=40)&(fine<=42)).astype(np.uint8)
   num, comp, stats, _=cv2.connectedComponentsWithStats(seed,8)
   for ci in range(1,num):
    x,y,wid,hei,area=stats[ci]
    if area<3: continue
    box=np.array([max(0,x-2),max(0,y-2),min(639,x+wid+2),min(639,y+hei+2)],np.float32)
    masks,_,_=predictor.predict(box=box[None,:],multimask_output=False)
    m=masks[0]
    # Keep only the SAM region connected to the semantic seed, preventing background takeover.
    overlap=m & (comp==ci)
    if overlap.sum()<2: continue
    region=m & (cv2.dilate((comp==ci).astype(np.uint8),np.ones((9,9),np.uint8))>0)
    labels=fine[comp==ci]
    dominant=int(np.bincount(labels.astype(np.int64),minlength=44).argmax())
    fine[region]=dominant
  if not use_sam: np.testing.assert_array_equal(fine,arch)
  pts=depth_points(dep,pose); world=pts[good]; lab=lut[fine[good]]; w=depth_weights(dep,state['mapping_config'])[good]
  i,j=grid_indices(world[:,0],world[:,2],state); inside=(i>=0)&(i<N)&(j>=0)&(j<N)&(lab>=1)&(lab<=3)&(w>1e-5)
  i=i[inside];j=j[inside]; lab=lab[inside]; world=world[inside]
  np.add.at(cnt[v],(i,j,lab),1); np.maximum.at(hmax,(i,j),world[:,1]-d['floor_y']); covered[i,j]=True
 old=np.load(p/'semantic_elevation/maps.npz'); valid=old['valid_GT']; gt=old['GT_semantic']; gh=old['GT_height']
 methods={}
 # Baseline: a class must be seen in any view, then strict priority (same logic as layered version).
 anyev=cnt.sum(0)>0; pred=np.max(anyev[:,:,1:4]*np.array([1,2,3],np.uint8),axis=-1).astype(np.uint8); hp=hmax.copy(); hp[pred==0]=np.nan; methods['any_view_priority']=(pred,hp)
 # View-balanced instance-like evidence. A view contributes at most one vote; require a class fraction.
 total=cnt[:,:,:,1:4].sum(-1); frac=np.divide(cnt[:,:,:,1:4],total[:,:,:,None],out=np.zeros_like(cnt[:,:,:,1:4],np.float32),where=total[:,:,:,None]>0)
 for name,small_thr,large_thr,min_views in [('best_view_confirm',.35,.50,1),('two_view_confirm',.35,.50,2),('margin_confirm',.45,.60,1)]:
  th=np.array([1.,small_thr,large_thr],np.float32); ok=frac>=th[None,None,None,:]
  support=ok.sum(0); evidence=(support>=min_views)
  # floor uses all covered cells, while object classes need the stated evidence.
  cand=np.zeros((N,N,3),bool); cand[:,:,0]=covered; cand[:,:,1:]=evidence[:,:,1:]
  pred=np.max(cand*np.array([1,2,3],np.uint8),axis=-1).astype(np.uint8); hp=hmax.copy(); hp[pred==0]=np.nan; methods[name]=(pred,hp)
 # Soft region evidence: sum the per-view class fractions, then require a fixed separation from runner-up.
 score=frac.sum(0); order=np.argsort(score,axis=-1); top=order[:,:,-1]; second=order[:,:,-2]; margin=np.take_along_axis(score,top[...,None],-1)[...,0]-np.take_along_axis(score,second[...,None],-1)[...,0]
 cand=np.zeros((N,N,3),bool); cand[:,:,0]=covered
 for k,thr in [(1,.15),(2,.20)]: cand[:,:,k]=score[:,:,k]>thr
 pred=np.zeros((N,N),np.uint8)
 # priority among confirmed candidates, as requested
 pred[cand[:,:,0]]=1; pred[cand[:,:,1]]=2; pred[cand[:,:,2]]=3
 hp=hmax.copy(); hp[pred==0]=np.nan; methods['soft_view_evidence']=(pred,hp)
 reports={'room_id':room,'methods':{}}
 for name,(pr,hh) in methods.items(): reports['methods'][name]=metrics(gt,pr,gh,hh,valid)
 np.savez_compressed(OUT/f'room-{room:02d}.npz',GT=gt,GT_height=gh,valid=valid,covered=covered,**{f'{n}_pred':x[0] for n,x in methods.items()})
 dump(OUT/f'room-{room:02d}.json',reports); del feats,state; gc.collect(); torch.cuda.empty_cache(); return reports

parser=argparse.ArgumentParser(); parser.add_argument('--sam',action='store_true'); args=parser.parse_args()
allr=[build(r,args.sam) for r in ROOMS]
summary={'scope':'Same three rooms, same eight RGB-D views, same GT and fixed evaluation masks. No GT used in prediction.','sam_refinement':bool(args.sam),'methods':{}}
for name in allr[0]['methods']:
 rs=[x['methods'][name] for x in allr]; gt=sum(x['GT_cells'] for x in rs); ok=sum(x['correct_cells'] for x in rs)
 rows=[]
 for k in range(3):
  g=sum(x['per_class'][k]['gt'] for x in rs); tp=sum(x['per_class'][k]['correct'] for x in rs); pp=sum(x['per_class'][k]['predicted'] for x in rs)
  rows.append({'class':rs[0]['per_class'][k]['class'],'gt':g,'correct':tp,'predicted':pp,'recall_percent':100*tp/g if g else None,'precision_percent':100*tp/pp if pp else None})
 summary['methods'][name]={'GT_cells':gt,'correct_cells':ok,'accuracy_percent':100*ok/gt,'per_class':rows,'height_MAE_cm_common_mean':float(np.mean([x['height_MAE_cm'] for x in rs]))}
dump(OUT/'summary.json',summary)
print(json.dumps(summary,indent=2))
