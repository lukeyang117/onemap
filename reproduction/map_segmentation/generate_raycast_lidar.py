"""Generate a physically ordered spinning LiDAR scan from Habitat raycasts."""
import argparse, json
from pathlib import Path
import numpy as np
import magnum as mn
import habitat_sim
from scipy.spatial.transform import Rotation, Slerp
from scene import ROOT, make_sim, set_pose, dump
from generate_habitat_lio_dataset import trajectory, HABITAT_TO_FASTLIO, imu_from_trajectory


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--room',type=int,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--frames',type=int,default=0); ap.add_argument('--rings',type=int,default=32); ap.add_argument('--azimuth',type=int,default=512); ap.add_argument('--dt',type=float,default=.1); args=ap.parse_args()
    src=ROOT/f'reproduction/map_segmentation/multi_room/room-{args.room:02d}'; design=json.loads((src/'design.json').read_text()); poses,dt=trajectory(design,.04,args.dt,smooth=True)
    if args.frames: poses=poses[:args.frames]
    args.output.mkdir(parents=True,exist_ok=True); (args.output/'lidar').mkdir(exist_ok=True)
    az=np.linspace(-np.pi,np.pi,args.azimuth,endpoint=False); elev=np.deg2rad(np.linspace(-15.,15.,args.rings))
    with make_sim(camera_height=.3,camera_forward=.3) as sim:
      for fi,p0 in enumerate(poses):
        p1=poses[min(fi+1,len(poses)-1)]; r0=Rotation.from_quat(p0['rotation_xyzw']);r1=Rotation.from_quat(p1['rotation_xyzw']);slerp=Slerp([0.,1.],Rotation.concatenate([r0,r1])); pts=[];rings=[];times=[]
        for ri,el in enumerate(elev):
          for ai,aa in enumerate(az):
            tau=ai/args.azimuth*dt; a=tau/dt; r=slerp(a); pos=(1-a)*np.asarray(p0['position'])+a*np.asarray(p1['position']); origin=pos+r.apply([0.,.3,-.3])
            pf=np.array([np.cos(el)*np.cos(aa),np.cos(el)*np.sin(aa),np.sin(el)])
            ph=HABITAT_TO_FASTLIO@pf; direction=r.apply(ph)
            hits=sim.cast_ray(habitat_sim.geo.Ray(mn.Vector3(origin),mn.Vector3(direction)))
            if not hits.has_hits(): continue
            hit=np.asarray(hits.hits[0].point,dtype=np.float64); local=r.inv().apply(hit-origin); pl=HABITAT_TO_FASTLIO@local
            if np.linalg.norm(pl)<.1: continue
            pts.append(pl);rings.append(ri);times.append(tau)
        np.savez_compressed(args.output/'lidar'/f'{fi:06d}.npz',timestamp=fi*dt,points_body=np.asarray(pts,np.float32),rings=np.asarray(rings,np.uint16),point_time=np.asarray(times,np.float32))
    np.savez_compressed(args.output/'imu.npz',**imu_from_trajectory(poses,dt))
    dump(args.output/'manifest.json',{'room':args.room,'frames':len(poses),'dt_s':dt,'sensor':'Habitat collision-mesh raycast spinning LiDAR','rings':args.rings,'azimuth_samples':args.azimuth,'point_time':'per-ray azimuth offset seconds','axis_transform_habitat_to_fastlio':HABITAT_TO_FASTLIO.tolist(),'lidar_to_imu_extrinsic_fastlio_m':[.3,0.,.3]})
    dump(args.output/'ground-truth-poses.json',{'poses':poses,'source':'Habitat ground truth for dataset audit only'})
    print(f'Generated raycast room {args.room}: {len(poses)} frames',flush=True)

if __name__=='__main__': main()
