"""Saved OneMap versus mesh GT: three-class semantic elevation maps."""
import json,csv,argparse
from pathlib import Path
import numpy as np
import torch
from scene import ROOT,dump,digest
from score_saved import grid_indices
from score_voxel_map import depth_points
from elevation_map import class_lookup,project_triangles,reduce_points

NAMES={1:'floor_stairs',2:'small_objects',3:'large_objects'}

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--experiment',type=Path,default=ROOT/'reproduction/map_segmentation/eight_view_map')
 parser.add_argument('--observed-only',action='store_true')
 args=parser.parse_args()
 source=args.experiment.resolve()
 out=source/'semantic_elevation';out.mkdir(exist_ok=True)
 d=json.loads((source/'design.json').read_text());state=torch.load(source/'augmented/saved-map.pt',map_location='cpu',weights_only=True)
 build=json.loads((source/'augmented/build-verification.json').read_text())
 assert build['map_updates']==8 and build['SAM_calls']==0
 assert digest(source/'augmented/saved-map.pt')==build['saved_map_sha256']
 lut=class_lookup(False);n=state['n_cells'];res=state['cell_size_m'];origin=np.array(state['origin_habitat']);floor=d['floor_y'];half=n*res/2
 mesh=np.load(source/'annotated-mesh.npz');tri=mesh['triangles'];fine_labels=mesh['labels'].astype(np.uint8)
 # World Habitat (x,y-up,z) -> upstream map planar coordinates, plus ground height.
 mapped=np.stack([-(tri[:,:,2]-origin[2])+half,-(tri[:,:,0]-origin[0])+half,tri[:,:,1]-floor],-1)
 gt_h,gt_s=project_triangles(mapped,lut[fine_labels],n,res)
 feature_grid=np.full((n,n),43,np.uint8);ids=state['indices'].numpy()
 feature_grid[ids[:,0],ids[:,1]]=(state['features']@state['text_features'].T).argmax(1).numpy()
 pred_h=np.full((n,n),np.nan);pred_s=np.zeros((n,n),np.uint8)
 independent_points=[];independent_labels=[];depth_coverage=np.zeros((n,n),bool)
 # The only prediction inputs are eight depths, poses and reloaded map features.
 for pose in d['poses']:
  dep=np.load(source/f"augmented/view-{pose['index']:02d}-geometry.npz")['depth']
  pts=depth_points(dep,pose);good=(dep>0)&np.isfinite(dep);pts=pts[good]
  i,j=grid_indices(pts[:,0],pts[:,2],state);inside=(i>=0)&(i<n)&(j>=0)&(j<n)
  pts=pts[inside];i=i[inside];j=j[inside];depth_coverage[i,j]=True;sem=lut[feature_grid[i,j]]
  mp=np.stack([-(pts[:,2]-origin[2])+half,-(pts[:,0]-origin[0])+half,pts[:,1]-floor],-1)
  h,s=reduce_points(mp,sem,n,res)
  pred_h=np.fmax(pred_h,h);pred_s=np.maximum(pred_s,s)
  independent_points.append(mp);independent_labels.append(sem)
 full_valid=gt_s>0
 valid=full_valid&depth_coverage if args.observed_only else full_valid
 correct=valid&(gt_s==pred_s)
 # Rows GT, columns prediction, zero column explicitly retains missing cells.
 cm=np.bincount(4*gt_s[valid].astype(int)+pred_s[valid],minlength=16).reshape(4,4)
 audit=np.zeros((4,4),np.int64);np.add.at(audit,(gt_s[valid],pred_s[valid]),1)
 np.testing.assert_array_equal(cm,audit)
 rows=[]
 for k,name in NAMES.items():
  mask=valid&(gt_s==k);total=int(mask.sum());tp=int(correct[mask].sum())
  rows.append({'class_id':k,'class_name':name,'GT_cells':total,'correct_cells':tp,'accuracy_percent':100*tp/total if total else None})
 matched=valid&np.isfinite(pred_h);height_error=pred_h[matched]-gt_h[matched]
 height_mae_cm=None if not matched.any() else 100*float(np.abs(height_error).mean())
 # Independently verify frame fusion against one concatenated reconstruction.
 h2,s2=reduce_points(np.concatenate(independent_points),np.concatenate(independent_labels),n,res)
 np.testing.assert_array_equal(s2,pred_s);np.testing.assert_allclose(h2,pred_h,equal_nan=True)
 assert np.isfinite(gt_h[valid]).all() and not np.isfinite(gt_h[~full_valid]).any()
 report={'semantic_accuracy_percent':100*float(correct.sum()/valid.sum()),'GT_cells':int(valid.sum()),'full_GT_cells':int(full_valid.sum()),'excluded_unobserved_cells':int((full_valid&~valid).sum()),'observed_only':args.observed_only,'room_id':d.get('room_id',22),'correct_cells':int(correct.sum()),
 'unmapped_GT_cells':int((valid&(pred_s==0)).sum()),'class_accuracy':rows,
 'height_MAE_cm':height_mae_cm,'height_matched_cells':int(matched.sum()),
 'grid_size_m':res,'mapping_RGBD_frames':8,'camera_height_m':.3,'camera_forward_offset_m':.3,'SAM_calls':0,
 'priority':'large_objects (3) > small_objects (2) > floor_stairs (1)',
 'height_rule':'Maximum height above the common floor datum, independently of selected semantic category',
 'GT_source':f"Exact planar cell clipping of independent room {d.get('room_id',22)} annotated triangles plus added object asset meshes. Ceiling triangles excluded before height/semantic reduction.",
 'prediction_source':'Eight raw depth reconstructions; semantics queried from losslessly reloaded original OneMap feature grid. Predicted ceiling/unknown/unmapped labels excluded before height/semantic reduction.',
 'category_mapping':{'floor_stairs':['floor','stairs'],'small_objects':['banana','tin_can','cracker_box'],'large_objects':'All remaining annotated semantic classes, including walls, furniture, objects and misc; excluding ceiling.','excluded':['ceiling','unmapped','unannotated']},
 'semantic_formula':'100 * number of GT-valid XY cells with identical three-class labels / number of GT-valid XY cells. Missing prediction counts as incorrect.',
 'height_formula':'100 * mean(abs(pred_max_height - GT_max_height)) on GT-valid cells with a valid predicted height; units cm. Missing cells excluded from MAE and included as semantic errors.',
 'domain':('GT cells with at least one input depth hit. Ceiling-filtered saved labels remain errors.' if args.observed_only else 'All labelled projected cells of this room, including unseen cells.')+' Predictions outside the GT domain are not scored.',
 'limitations':['Semantic categories are mapped after upstream OneMap feature fusion; original OneMap cannot separate classes by height within one planar cell.','Priority can hide small objects sharing an XY cell with a large object.','Small objects are the three added asset classes, not a physical size threshold.',('No true stair surfaces in this room; floor_stairs measures floor only.' if not np.any(fine_labels==15) else 'This room contains stair annotations.'),'For predicted height, ceiling removal relies on predicted semantics, not GT masks; a ceiling misclassified as another class can still inflate predicted height.'],
 'saved_map_sha256':build['saved_map_sha256'],'checks':{'independent_confusion_equal':True,'frame_fusion_equal_to_single_reduction':True,'eight_mapping_frames':True,'no_GT_masks_used_for_prediction':True}}
 dump(out/'results.json',report)
 np.savez_compressed(out/'maps.npz',GT_height=gt_h,predict_height=pred_h,GT_semantic=gt_s,predict_semantic=pred_s,confusion_matrix=cm,valid_GT=valid,full_valid_GT=full_valid,raw_depth_coverage=depth_coverage,cell_size_m=res,origin_habitat=origin,floor_datum_y=floor)
 with (out/'class-accuracy.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (out/'per-cell-comparison.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['grid_i','grid_j','world_X_m','world_Z_m','GT_max_height_above_floor_m','predict_max_height_above_floor_m','GT_class','predict_class','semantic_correct','height_error_m'])
  names={0:'missing',**NAMES}
  for i,j in np.argwhere(valid):
   a=float(gt_h[i,j]);b=float(pred_h[i,j]);g=int(gt_s[i,j]);q=int(pred_s[i,j])
   w.writerow([i,j,origin[0]-(j+.5-n//2)*res,origin[2]-(i+.5-n//2)*res,a,b if np.isfinite(b) else '',names[g],names[q],int(g==q),b-a if np.isfinite(b) else ''])
 print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
