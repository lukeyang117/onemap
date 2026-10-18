"""Convert recorded FAST-LIO odometry to the eight OneMap evaluation poses."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores,get_typestore

L=np.array([[0.,0.,-1.],[-1.,0.,0.],[0.,1.,0.]])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--room',type=int,required=True);ap.add_argument('--bag',type=Path,required=True);ap.add_argument('--truth',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 design=json.loads((Path('reproduction/map_segmentation/multi_room')/f'room-{args.room:02d}'/'design.json').read_text());truth=json.loads(args.truth.read_text())['poses'];G=np.array([p['position'] for p in truth]);T=get_typestore(Stores.ROS2_HUMBLE);raw=[]
 with Reader(args.bag) as rd:
  for c,t,b in rd.messages():
   m=T.deserialize_cdr(b,c.msgtype);p=m.pose.pose.position;q=m.pose.pose.orientation;raw.append({'position':[p.x,p.y,p.z],'rotation_xyzw':[q.x,q.y,q.z,q.w]})
 E=np.array([x['position'] for x in raw]);idx=np.rint(np.linspace(0,len(raw)-1,len(truth))).astype(int);H=(E[idx]-E[idx].mean(0)).T@(G-G.mean(0));u,_,vt=np.linalg.svd(H);A=vt.T@u.T
 if np.linalg.det(A)<0:vt[-1]*=-1;A=vt.T@u.T
 t=G.mean(0)-A@E[idx].mean(0);poses=[]
 for k,d in enumerate([p for p in design['poses'] if p['split']=='mapping']):
  j=int(np.argmin(np.linalg.norm(G-np.asarray(d['position']),axis=1)));q=raw[idx[j]];poses.append({'index':k,'position':(A@np.asarray(q['position'])+t).tolist(),'rotation_xyzw':Rotation.from_matrix(A@Rotation.from_quat(q['rotation_xyzw']).as_matrix()@L).as_quat().tolist(),'split':'mapping','source':'FAST-LIO odometry'})
 args.output.write_text(json.dumps({'poses':poses,'source':'FAST-LIO odometry; rigid world-frame alignment for scoring'},indent=2)+'\n');print(json.dumps({'room':args.room,'odom_frames':len(raw),'output':str(args.output)}))
if __name__=='__main__':main()
