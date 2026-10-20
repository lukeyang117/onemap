"""Controlled diagnostics of small-object losses, without changing official maps.
1 full-frame dense pixel scores; 2 actual single/cumulative OneMap; 3 no-gradient
and finer-grid ablations; 4 GT-semantic oracle mean-fusion versus priority;
5 GT-assisted zoom sanity check (diagnostic only, never a deployable score).
"""
import json,csv,gc
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from scene import ROOT,CLASSES,dump,pose_matrix
from score_saved import grid_indices
from score_voxel_map import depth_points

BASE=ROOT/'reproduction/map_segmentation/multi_room';OUT=BASE/'small_object_analysis'
OUT.mkdir(exist_ok=True)
K=np.array([[320,0,320],[0,320,320],[0,0,1]])

def newmap(settings,dim=768):
 from mapping.feature_map import OneMap
 m=OneMap(dim,SimpleNamespace(**settings),map_device='cpu');m.set_camera_matrix(K);return m

def labels_at(mapper,xy,text=None):
 i=np.floor(-(xy[:,1]-mapper._origin[2])/mapper.cell_size).astype(int)+mapper.n_cells//2
 j=np.floor(-(xy[:,0]-mapper._origin[0])/mapper.cell_size).astype(int)+mapper.n_cells//2
 inside=(i>=0)&(i<mapper.n_cells)&(j>=0)&(j<mapper.n_cells)
 pred=np.full(len(xy),43,np.uint8)
 ii=i[inside];jj=j[inside];feat=mapper.feature_map[ii,jj]
 logits=feat@text.T if text is not None else feat
 lab=logits.argmax(1).numpy().astype(np.uint8);lab[mapper.confidence_map[ii,jj].numpy()<=0]=43
 pred[inside]=lab;return pred

def small(lab):return (lab>=40)&(lab<=42)

def dense_labels(feat,text):
 scores=torch.einsum('chw,kc->khw',feat,text.to(feat.device))
 scores=F.interpolate(scores[None],size=(640,640),mode='bilinear',align_corners=False)[0]
 return scores.argmax(0).cpu().numpy().astype(np.uint8)

def depth_weights(depth,settings):
 dep=torch.as_tensor(depth.copy(),device='cuda');smooth=-F.max_pool2d(-dep[None],11,padding=5,stride=1)[0]
 dx=torch.gradient(smooth,dim=1)[0]/.5;dy=torch.gradient(smooth,dim=0)[0]/.5
 grad=F.max_pool2d(torch.sqrt(dx**2+dy**2)[None],11,padding=5,stride=1)[0]
 return ((1-torch.tanh(grad*settings['gradient_factor']))*torch.exp(-((settings['optimal_object_distance']-dep)/settings['optimal_object_factor'])**2/3)).cpu().numpy()

def run():
 from vision_models.clip_dense import ClipModel
 model=ClipModel(str(ROOT/'weights/clip.pth'),jetson=False).eval()
 rooms=json.loads((BASE/'selection.json').read_text())['selected_rooms'];all_frames=[];object_rows=[];room_rows=[]
 for room in rooms:
  print('Starting room',room,flush=True)
  p=BASE/f'room-{room:02d}';out=OUT/f'room-{room:02d}';out.mkdir(exist_ok=True)
  d=json.loads((p/'design.json').read_text());state=torch.load(p/'augmented/saved-map.pt',map_location='cpu',weights_only=True);text=state['text_features'];settings=state['mapping_config'];maps=np.load(p/'semantic_elevation/maps.npz')
  gtmask=maps['valid_GT']&(maps['GT_semantic']==2);ij=np.argwhere(gtmask);origin=np.array(d['origin']);xy=np.stack([origin[0]-(ij[:,1]+.5-300)*.1,origin[2]-(ij[:,0]+.5-300)*.1],-1) if len(ij) else np.empty((0,2))
  # Pixel/feature evidence and reproducible full-size feature cache.
  frames=[];obj_data={o['semantic_id']:{'item':o,'points':[],'frames':[]} for o in d['objects']}
  class_counts=np.zeros((5,600,600),np.int64)
  for pose in d['poses']:
   idx=pose['index'];stem=p/f'augmented/view-{idx:02d}';rgb=np.array(Image.open(str(stem)+'-rgb.png'));geo=np.load(str(stem)+'-geometry.npz');dep=geo['depth'].copy();inst=geo['instance'];truth=np.array(Image.open(str(stem)+'-gt.png'))
   with torch.inference_mode():feat=model.get_image_features(rgb.transpose(2,0,1))[0];pred=dense_labels(feat,text)
   shape=list(feat.shape);feat=feat.cpu();weights=depth_weights(dep,settings);world=depth_points(dep,pose)
   points=world[(dep>0)&np.isfinite(dep)];tr=truth[(dep>0)&np.isfinite(dep)]
   a,b=grid_indices(points[:,0],points[:,2],state);keep=(a>=0)&(a<600)&(b>=0)&(b<600);a=a[keep];b=b[keep];tr=tr[keep]
   category=np.full(len(tr),3,np.uint8);category[np.isin(tr,[1,15])]=1;category[np.isin(tr,[40,41,42])]=2;category[tr==16]=4;category[tr==255]=0;np.add.at(class_counts,(category,a,b),1)
   for sid,data in obj_data.items():
    mask=(inst==sid)&(dep>0)&np.isfinite(dep);count=int(mask.sum())
    if not count:continue
    y,x=np.where(mask);hist=np.bincount(pred[mask],minlength=44)
    ring=(cv2.dilate(mask.astype(np.uint8),np.ones((31,31),np.uint8))>0)&~mask&(truth==1)
    occupancy=cv2.resize(mask.astype(np.float32),(shape[2],shape[1]),interpolation=cv2.INTER_AREA)
    row={'room_id':room,'view':idx,'semantic_id':sid,'asset':data['item']['asset'],'size_cm':int(round(data['item']['longest_side_m']*100)),
     'visible_pixels':count,'bbox_width_px':int(x.max()-x.min()+1),'bbox_height_px':int(y.max()-y.min()+1),'median_depth_m':float(np.median(dep[mask])),
     'any_small_correct_pixels':int(small(pred[mask]).sum()),'exact_asset_correct_pixels':int((pred[mask]==data['item']['class_id']).sum()),
     'max_nominal_feature_bin_object_fraction':float(occupancy.max()),'mean_object_depth_weight':float(weights[mask].mean()),'mean_nearby_floor_depth_weight':float(weights[ring].mean()) if ring.any() else None,
     'fraction_object_weights_rejected':float((weights[mask]<=1e-5).mean()),'predicted_class_histogram':hist.tolist()}
    data['frames'].append(row);all_frames.append(row);data['points'].append(world[mask]);
   frames.append({'pose':pose,'rgb':rgb,'depth':dep,'instance':inst,'truth':truth,'features':feat,'prediction':pred})
   Image.fromarray(pred).save(out/f'view-{idx:02d}-frontend-classes.png')
  torch.save([f['features'] for f in frames],out/'cached-dense-features.pt')
  # Fixed unique 2cm observed object-surface samples for resolution/weight ablations.
  for sid,data in obj_data.items():
   pts=np.concatenate(data['points']);_,first=np.unique(np.floor(pts/.02).astype(np.int64),axis=0,return_index=True);data['samples']=pts[first][:,[0,2]]
  query_points=np.concatenate([z['samples'] for z in obj_data.values()]);splits=np.cumsum([len(z['samples']) for z in obj_data.values()])
  mode_results={};single_any=np.zeros(len(xy),bool);cumulative_any=np.zeros(len(xy),bool);history=[]
  for mode in ['baseline_replay','gradient_weight_disabled','grid_5cm','GT_semantic_oracle']:
   conf=dict(settings);dim=768
   if mode=='gradient_weight_disabled':conf['gradient_factor']=0.
   if mode=='grid_5cm':conf.update({'size':30.,'blur_kernel_size':0.})
   if mode=='GT_semantic_oracle':dim=43
   mapper=newmap(conf,dim);mapper._origin=origin
   for f in frames:
    tf=pose_matrix(f['pose'],origin,.3)
    if mode=='GT_semantic_oracle':
     truth=f['truth'];onehot=F.one_hot(torch.as_tensor(np.where(truth==255,43,truth).astype(np.int64),device='cuda'),44)[...,:43].permute(2,0,1).float();values=onehot
    else:values=f['features'].cuda()
    with torch.inference_mode():mapper.update(values,f['depth'].copy(),tf)
    if mode=='baseline_replay':
     lab=labels_at(mapper,xy,text) if len(xy) else np.empty(0,np.uint8);correct=small(lab);cumulative_any|=correct;history.append({'view_count':f['pose']['index']+1,'small_GT_cells_correct':int(correct.sum()),'labels':lab.tolist()})
     single=newmap(conf);single._origin=origin
     with torch.inference_mode():single.update(values,f['depth'].copy(),tf)
     single_labels=labels_at(single,xy,text) if len(xy) else np.empty(0,np.uint8);single_any|=small(single_labels)
     del single;gc.collect()
    del values
   labs=labels_at(mapper,query_points,None if mode=='GT_semantic_oracle' else text)
   for data,part in zip(obj_data.values(),np.split(labs,splits[:-1])):data.setdefault('surface_query',{})[mode]={'correct':int(small(part).sum()),'total':len(part),'predicted_histogram':np.bincount(part,minlength=44).tolist()}
   cell_lab=labels_at(mapper,xy,None if mode=='GT_semantic_oracle' else text) if len(xy) else np.empty(0,np.uint8)
   mode_results[mode]={'original_10cm_small_GT_cells':len(xy),'small_GT_cell_center_labels':cell_lab.tolist(),'small_GT_cells_correct':(None if mode=='grid_5cm' else int(small(cell_lab).sum())),'fixed_object_surface_correct':int(small(labs).sum()),'fixed_object_surface_total':len(labs)}
   if mode=='baseline_replay':
    ids=state['indices'].long();active=(mapper.confidence_map>0).nonzero();np.testing.assert_array_equal(active.numpy(),ids.numpy())
    torch.testing.assert_close(mapper.feature_map[ids[:,0],ids[:,1]],state['features'],rtol=1e-4,atol=1e-5)
    expected=maps['predict_semantic'][gtmask]==2;np.testing.assert_array_equal(small(cell_lab),expected)
   del mapper;gc.collect();torch.cuda.empty_cache()
   print('Room',room,mode,mode_results[mode],flush=True)
  # GT semantic priority reconstruction is a diagnostic visibility upper-bound.
  priority=np.zeros((600,600),np.uint8)
  for f in frames:
   good=(f['depth']>0)&np.isfinite(f['depth']);points=depth_points(f['depth'],f['pose'])[good];truth=f['truth'][good]
   a,b=grid_indices(points[:,0],points[:,2],state);inside=(a>=0)&(a<600)&(b>=0)&(b<600);a=a[inside];b=b[inside];tr=truth[inside]
   cat=np.full(len(tr),3,np.uint8);cat[np.isin(tr,[1,15])]=1;cat[np.isin(tr,[40,41,42])]=2;cat[np.isin(tr,[16,255])]=0;np.maximum.at(priority,(a,b),cat)
  final=maps['predict_semantic'][gtmask]==2
  counts=class_counts[:,ij[:,0],ij[:,1]] if len(ij) else np.empty((5,0))
  room_record={'room_id':room,'GT_small_cells':len(ij),'baseline_final_small_correct':int(final.sum()),'ever_single_view_small_correct':int(single_any.sum()),'ever_cumulative_small_correct':int(cumulative_any.sum()),'single_view_ever_correct_but_final_wrong':int((single_any&~final).sum()),'cumulative_ever_correct_but_final_wrong':int((cumulative_any&~final).sum()),
   'GT_priority_oracle_small_correct':int((priority[gtmask]==2).sum()),'per_cell_raw_pixel_category_counts':counts.T.tolist(),'category_count_columns':['unknown','floor_stairs','small_objects','large_objects','ceiling'],'modes':mode_results,'cumulative_history':history,'dense_feature_shape':shape,'baseline_replay_matches_saved_features':True}
  dump(out/'map-stage-results.json',room_record);room_rows.append(room_record)
  # GT-assisted tight crop/zoom, best visible view chosen by GT area only.
  for sid,data in obj_data.items():
   best=max(data['frames'],key=lambda r:r['visible_pixels']);f=frames[best['view']];mask=f['instance']==sid;y,x=np.where(mask);side=max(x.max()-x.min()+1,y.max()-y.min()+1)*1.5;cx=(x.min()+x.max())/2;cy=(y.min()+y.max())/2
   x0=max(0,int(cx-side/2));x1=min(640,int(np.ceil(cx+side/2))+1);y0=max(0,int(cy-side/2));y1=min(640,int(np.ceil(cy+side/2))+1)
   rgb=np.asarray(Image.fromarray(f['rgb'][y0:y1,x0:x1]).resize((640,640),Image.Resampling.BILINEAR));zoom_mask=np.asarray(Image.fromarray(mask[y0:y1,x0:x1]).resize((640,640),Image.Resampling.NEAREST)).astype(bool)
   with torch.inference_mode():feat=model.get_image_features(rgb.transpose(2,0,1))[0];pred=dense_labels(feat,text)
   zoom_recall=float(small(pred[zoom_mask]).mean());full_recall=best['any_small_correct_pixels']/best['visible_pixels']
   item=data['item'];row={'room_id':room,'semantic_id':sid,'asset':item['asset'],'size_cm':int(round(item['longest_side_m']*100)),
    'total_visible_input_pixels':sum(z['visible_pixels'] for z in data['frames']),'visible_views':len(data['frames']),'best_view':best['view'],'best_view_visible_pixels':best['visible_pixels'],'best_bbox_width_px':best['bbox_width_px'],'best_bbox_height_px':best['bbox_height_px'],
    'full_frame_pixel_recall_percent':100*sum(z['any_small_correct_pixels'] for z in data['frames'])/sum(z['visible_pixels'] for z in data['frames']),
    'best_view_pixel_recall_percent':100*full_recall,'GT_zoom_pixel_recall_percent':100*zoom_recall,
    'best_feature_bin_object_fraction':best['max_nominal_feature_bin_object_fraction'],'surface_queries':data['surface_query']}
   object_rows.append(row)
   Image.fromarray(rgb).save(out/f'object-{sid}-GT-assisted-zoom.png');Image.fromarray(pred).save(out/f'object-{sid}-zoom-classes.png')
  dump(out/'object-results.json',[r for r in object_rows if r['room_id']==room]);del frames,state;gc.collect();torch.cuda.empty_cache()
  print('Room complete',room,flush=True)
 dump(OUT/'per-view-object-evidence.json',all_frames);dump(OUT/'per-object-results.json',object_rows);dump(OUT/'per-room-results.json',room_rows)
 print('All diagnostics completed',flush=True)

if __name__=='__main__':run()
