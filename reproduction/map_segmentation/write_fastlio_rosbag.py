"""Convert exported Habitat LiDAR/IMU NPZ files into a ROS1 bag.

Requires the pure-Python ``rosbags`` package; it does not require ROS to write
the bag. FAST-LIO itself still requires a ROS runtime and its configured topic,
extrinsic, point-time and IMU conventions.
"""
import argparse, json, struct
from pathlib import Path
import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore


def stamp(t, Time):
    sec=int(t); return Time(sec, int(round((t-sec)*1e9)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    ts=get_typestore(Stores.ROS1_NOETIC);T=ts.types
    Header=T['std_msgs/msg/Header'];Time=T['builtin_interfaces/msg/Time'];PF=T['sensor_msgs/msg/PointField'];PC=T['sensor_msgs/msg/PointCloud2'];Imu=T['sensor_msgs/msg/Imu'];Vec=T['geometry_msgs/msg/Vector3'];Quat=T['geometry_msgs/msg/Quaternion']
    lidar_conn=None;imu_conn=None;imu=np.load(args.input/'imu.npz');files=sorted((args.input/'lidar').glob('*.npz'));args.output.parent.mkdir(parents=True,exist_ok=True)
    with Writer(args.output) as writer:
        lidar_conn=writer.add_connection('/fastlio/lidar','sensor_msgs/msg/PointCloud2',typestore=ts);imu_conn=writer.add_connection('/fastlio/imu','sensor_msgs/msg/Imu',typestore=ts)
        for k,f in enumerate(files):
            d=np.load(f);t=float(d['timestamp']);pts=np.asarray(d['points_body'],np.float32);n=len(pts);times=np.linspace(0,.1,n,endpoint=False,dtype=np.float32);arr=np.empty((n,4),np.float32);arr[:,:3]=pts;arr[:,3]=times
            msg=PC(Header(0,stamp(t,Time),'lidar'),1,n,[PF('x',0,7,1),PF('y',4,7,1),PF('z',8,7,1),PF('time',12,7,1)],False,16,16*n,arr.view(np.uint8).reshape(-1),True)
            writer.write(lidar_conn,int(t*1e9),ts.serialize_ros1(msg,msg.__msgtype__))
            a=np.asarray(imu['linear_acceleration'][k]);w=np.asarray(imu['angular_velocity'][k]);m=Imu(Header(0,stamp(t,Time),'imu'),Quat(0.,0.,0.,1.),np.zeros(9),Vec(*w),np.zeros(9),Vec(*a),np.zeros(9));writer.write(imu_conn,int(t*1e9),ts.serialize_ros1(m,m.__msgtype__))
    manifest={'input':str(args.input),'output':str(args.output),'frames':len(files),'topics':{'lidar':'/fastlio/lidar','imu':'/fastlio/imu'},'point_fields':['x','y','z','time'],'frame_id':'lidar','ros_version':'ros1_noetic','fastlio_extrinsic_required':True,'fastlio_runtime_available':False}
    (args.output.with_suffix('.json')).write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
