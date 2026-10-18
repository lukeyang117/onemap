"""Generate synchronized synthetic LiDAR/IMU data from a Habitat trajectory.

Habitat's Python bindings used by this project expose RGB/depth/semantic/audio
but not LiDAR or IMU sensor types. This exporter therefore derives a sparse
LiDAR scan from the rendered depth image and IMU samples from the simulator
ground-truth trajectory. The output is intended for a ROS/FAST-LIO converter;
it is not itself a FAST-LIO estimate.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from PIL import Image
from scene import ROOT, make_sim, set_pose, dump


def interpolate_pose(a, b, alpha):
    ra, rb = Rotation.from_quat(a['rotation_xyzw']), Rotation.from_quat(b['rotation_xyzw'])
    slerp = Slerp([0., 1.], Rotation.concatenate([ra, rb]))
    q = slerp([alpha])[0].as_quat()
    p = (1-alpha)*np.asarray(a['position']) + alpha*np.asarray(b['position'])
    return {'position': p.tolist(), 'rotation_xyzw': q.tolist(), 'index': -1, 'split': 'sensor'}


def trajectory(design, spacing=.12, dt=.1):
    base=[p for p in design['poses'] if p.get('split') == 'mapping']
    out=[]
    for a,b in zip(base[:-1],base[1:]):
        distance=np.linalg.norm(np.asarray(b['position'])-np.asarray(a['position']))
        n=max(1,int(math.ceil(distance/spacing)))
        for k in range(n): out.append(interpolate_pose(a,b,k/n))
    out.append(dict(base[-1]))
    for i,p in enumerate(out): p['index']=i; p['timestamp']=i*dt
    return out,dt


def depth_to_lidar(depth, pose, camera_height, camera_forward, rows=32, cols=512):
    h,w=depth.shape; yy=np.linspace(2,h-2,rows).astype(int);xx=np.linspace(2,w-2,cols).astype(int)
    y,x=np.meshgrid(yy,xx,indexing='ij');d=depth[y,x]
    valid=(d>0)&np.isfinite(d)
    cam=np.stack([(x-320)*d/320., -(y-320)*d/320., -d],-1)[valid]
    R=Rotation.from_quat(pose['rotation_xyzw']).as_matrix();
    camera_world=np.asarray(pose['position'])+R@np.array([0.,camera_height,-camera_forward])
    world=cam@R.T+camera_world
    body=np.asarray(pose['position']);points_body=(world-body)@R
    return points_body.astype(np.float32),valid


def imu_from_trajectory(poses, dt):
    pos=np.asarray([p['position'] for p in poses],np.float64)
    rot=Rotation.from_quat(np.asarray([p['rotation_xyzw'] for p in poses]))
    vel=np.gradient(pos,dt,axis=0);acc=np.gradient(vel,dt,axis=0)
    gravity=np.array([0.,-9.81,0.]); lin=[];ang=[]
    for i,r in enumerate(rot):
        j=max(0,min(len(rot)-1,i+1)); delta=r.inv()*rot[j]; rv=delta.as_rotvec()/max(dt,1e-9)
        lin.append(r.inv().apply(acc[i]-gravity));ang.append(rv)
    return {'timestamp':np.arange(len(poses))*dt,'linear_acceleration':np.asarray(lin,np.float32),'angular_velocity':np.asarray(ang,np.float32)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--room',type=int,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--spacing',type=float,default=.12);ap.add_argument('--dt',type=float,default=.1);ap.add_argument('--fastlio',action='store_true',help='use dense 4 cm/frame motion sampling for FAST-LIO replay');args=ap.parse_args()
    if args.fastlio:
        args.spacing=min(args.spacing, .04)
    src=ROOT/f'reproduction/map_segmentation/multi_room/room-{args.room:02d}';design=json.loads((src/'design.json').read_text());poses,dt=trajectory(design,args.spacing,args.dt);args.output.mkdir(parents=True,exist_ok=True);(args.output/'lidar').mkdir(exist_ok=True)
    with make_sim(camera_height=.3,camera_forward=.3) as sim:
        for p in poses:
            set_pose(sim,p);obs=sim.get_sensor_observations();depth=np.asarray(obs['depth']);points,valid=depth_to_lidar(depth,p,.3,.3);np.savez_compressed(args.output/'lidar'/f'{p["index"]:06d}.npz',timestamp=p['timestamp'],points_body=points)
    imu=imu_from_trajectory(poses,dt);np.savez_compressed(args.output/'imu.npz',**imu);dump(args.output/'ground-truth-poses.json',{'poses':poses,'camera':design['camera'],'source':'Habitat simulator ground truth'});dump(args.output/'manifest.json',{'room':args.room,'frames':len(poses),'dt_s':dt,'motion_spacing_m':args.spacing,'motion_profile':'dense piecewise SLERP/linear interpolation' if args.fastlio else 'piecewise SLERP/linear interpolation','lidar_points_per_frame':'variable from valid depth rays','lidar_frame':'synthetic lidar frame coincident with Habitat body frame','imu_frame':'Habitat body frame','fastlio_ready':False,'reason':'ROS message conversion and FAST-LIO replay are separate steps'})
    print(f'Generated room {args.room}: {len(poses)} synchronized frames at {args.output}',flush=True)

if __name__=='__main__':main()
