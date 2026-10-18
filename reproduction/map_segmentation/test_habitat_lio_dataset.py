import numpy as np
from generate_habitat_lio_dataset import imu_from_trajectory, depth_to_lidar

def test_imu_gravity_and_shape():
 poses=[{'position':[i*.1,0,0],'rotation_xyzw':[0,0,0,1]} for i in range(4)]
 out=imu_from_trajectory(poses,.1)
 assert out['linear_acceleration'].shape==(4,3)
 assert out['angular_velocity'].shape==(4,3)

def test_depth_lidar_projection():
 depth=np.ones((640,640),np.float32)
 pose={'position':[0,0,0],'rotation_xyzw':[0,0,0,1]}
 points,valid=depth_to_lidar(depth,pose,.3,.3,rows=4,cols=8)
 assert points.shape==(32,3)
 assert np.isfinite(points).all()
