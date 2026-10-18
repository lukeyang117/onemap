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

# Habitat agent/camera axes are (right, up, backward).  FAST-LIO's usual
# Velodyne convention is (forward, left, up).  Keeping this transform explicit
# avoids feeding a valid PointCloud2 with an internally inconsistent frame.
HABITAT_TO_FASTLIO = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]], dtype=np.float64)


def interpolate_pose(a, b, alpha):
    ra, rb = Rotation.from_quat(a['rotation_xyzw']), Rotation.from_quat(b['rotation_xyzw'])
    slerp = Slerp([0., 1.], Rotation.concatenate([ra, rb]))
    q = slerp([alpha])[0].as_quat()
    p = (1-alpha)*np.asarray(a['position']) + alpha*np.asarray(b['position'])
    return {'position': p.tolist(), 'rotation_xyzw': q.tolist(), 'index': -1, 'split': 'sensor'}


def trajectory(design, spacing=.12, dt=.1, smooth=False):
    base=[p for p in design['poses'] if p.get('split') == 'mapping']
    out=[]
    for a,b in zip(base[:-1],base[1:]):
        distance=np.linalg.norm(np.asarray(b['position'])-np.asarray(a['position']))
        n=max(1,int(math.ceil(distance/spacing)))
        for k in range(n):
            alpha=k/n
            if smooth:
                alpha=alpha*alpha*(3.0-2.0*alpha)
            out.append(interpolate_pose(a,b,alpha))
    out.append(dict(base[-1]))
    if smooth:
        # FAST-LIO initializes velocity and biases from a stationary interval.
        # Hold the first view long enough for that initialization to complete.
        out=[dict(base[0]) for _ in range(max(15, int(round(1.5/dt))))]+out
    for i,p in enumerate(out): p['index']=i; p['timestamp']=i*dt
    return out,dt


def depth_to_lidar(depth, pose, camera_height, camera_forward, rows=32, cols=512):
    h,w=depth.shape; yy=np.linspace(2,h-2,rows).astype(int);xx=np.linspace(2,w-2,cols).astype(int)
    y,x=np.meshgrid(yy,xx,indexing='ij');d=depth[y,x]
    valid=(d>0)&np.isfinite(d)
    ring_ids=np.broadcast_to(np.arange(rows, dtype=np.uint16)[:,None], d.shape)[valid]
    cam=np.stack([(x-320)*d/320., -(y-320)*d/320., -d],-1)[valid]
    R=Rotation.from_quat(pose['rotation_xyzw']).as_matrix();
    camera_world=np.asarray(pose['position'])+R@np.array([0.,camera_height,-camera_forward])
    world=cam@R.T+camera_world
    # A LiDAR mounted at the camera origin.  Return coordinates in the
    # FAST-LIO sensor frame; the IMU-to-LiDAR translation is written to the
    # launch/config file and is therefore applied by FAST-LIO exactly once.
    points_sensor_h=(world-camera_world)@R
    points_sensor_fast=points_sensor_h@HABITAT_TO_FASTLIO.T
    return points_sensor_fast.astype(np.float32),valid,ring_ids


def imu_from_trajectory(poses, dt):
    pos=np.asarray([p['position'] for p in poses],np.float64)
    rot=Rotation.from_quat(np.asarray([p['rotation_xyzw'] for p in poses]))
    vel=np.gradient(pos,dt,axis=0);acc=np.gradient(vel,dt,axis=0)
    gravity=np.array([0.,-9.81,0.]); lin=[];ang=[]
    for i,r in enumerate(rot):
        j=max(0,min(len(rot)-1,i+1)); delta=r.inv()*rot[j]; rv=delta.as_rotvec()/max(dt,1e-9)
        lin_h=r.inv().apply(acc[i]-gravity)
        # Rotate vector measurements into FAST-LIO's sensor/IMU axes.
        lin.append(HABITAT_TO_FASTLIO@lin_h)
        ang.append(HABITAT_TO_FASTLIO@rv)
    return {'timestamp':np.arange(len(poses))*dt,'linear_acceleration':np.asarray(lin,np.float32),'angular_velocity':np.asarray(ang,np.float32)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--room',type=int,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--spacing',type=float,default=.12);ap.add_argument('--dt',type=float,default=.1);ap.add_argument('--fastlio',action='store_true',help='use dense 4 cm/frame motion sampling for FAST-LIO replay');ap.add_argument('--lidar360',action='store_true',help='simulate four synchronized 90-degree LiDAR sectors');args=ap.parse_args()
    if args.fastlio:
        args.spacing=min(args.spacing, .04)
    src=ROOT/f'reproduction/map_segmentation/multi_room/room-{args.room:02d}';design=json.loads((src/'design.json').read_text());poses,dt=trajectory(design,args.spacing,args.dt,smooth=args.fastlio);args.output.mkdir(parents=True,exist_ok=True);(args.output/'lidar').mkdir(exist_ok=True)
    with make_sim(camera_height=.3,camera_forward=.3) as sim:
        for p in poses:
            if not args.lidar360:
                set_pose(sim,p);obs=sim.get_sensor_observations();depth=np.asarray(obs['depth']);points,valid,rings=depth_to_lidar(depth,p,.3,.3)
            else:
                # Four 90-degree sectors provide scan overlap in rooms where a
                # single 90-degree RGB-D view sees mostly a nearby wall.  Eight
                # vertical rows per sector keep exactly 32 FAST-LIO rings.
                base=Rotation.from_quat(p['rotation_xyzw']); chunks=[]; ring_chunks=[]
                for yaw in (0.,90.,180.,270.):
                    q=(base*Rotation.from_euler('y',yaw,degrees=True)).as_quat(); pp=dict(p); pp['rotation_xyzw']=q.tolist()
                    set_pose(sim,pp); depth=np.asarray(sim.get_sensor_observations()['depth']); part,_,rr=depth_to_lidar(depth,pp,.3,.3,rows=8,cols=512); chunks.append(part); ring_chunks.append(rr)
                # Depth validity is per pixel, so each sector has a variable
                # number of returns.  Keep sector-major order; the ROS2 writer
                # assigns deterministic synthetic rings to the resulting scan.
                points=np.concatenate(chunks,axis=0); rings=np.concatenate(ring_chunks,axis=0)
                order=np.lexsort((np.arctan2(points[:,1],points[:,0]), rings))
                points=points[order]; rings=rings[order]
            np.savez_compressed(args.output/'lidar'/f'{p["index"]:06d}.npz',timestamp=p['timestamp'],points_body=points,rings=rings)
    imu=imu_from_trajectory(poses,dt);np.savez_compressed(args.output/'imu.npz',**imu);dump(args.output/'ground-truth-poses.json',{'poses':poses,'camera':design['camera'],'source':'Habitat simulator ground truth'});dump(args.output/'manifest.json',{'room':args.room,'frames':len(poses),'dt_s':dt,'motion_spacing_m':args.spacing,'motion_profile':'stationary initialization + smoothstep piecewise SLERP interpolation' if args.fastlio else 'piecewise SLERP/linear interpolation','axis_transform_habitat_to_fastlio':HABITAT_TO_FASTLIO.tolist(),'lidar_origin':'camera optical center','lidar_sectors':4 if args.lidar360 else 1,'lidar_to_imu_extrinsic_fastlio_m':[0.3,0.0,0.3],'lidar_points_per_frame':'variable from valid depth rays','lidar_frame':'FAST-LIO Velodyne convention (forward,left,up)','imu_frame':'FAST-LIO body convention (forward,left,up)','fastlio_ready':False,'reason':'ROS message conversion and FAST-LIO replay are separate steps'})
    print(f'Generated room {args.room}: {len(poses)} synchronized frames at {args.output}',flush=True)

if __name__=='__main__':main()
