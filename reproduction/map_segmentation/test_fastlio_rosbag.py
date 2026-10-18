import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores,get_typestore

def test_rosbag_has_lidar_and_imu(tmp_path):
    from write_fastlio_rosbag import main
    import sys
    inp=tmp_path/'in';(inp/'lidar').mkdir(parents=True)
    np.savez(inp/'lidar/000000.npz',timestamp=0.,points_body=np.zeros((3,3),np.float32))
    np.savez(inp/'imu.npz',timestamp=np.array([0.]),linear_acceleration=np.zeros((1,3)),angular_velocity=np.zeros((1,3)))
    out=tmp_path/'x.bag';sys.argv=['write_fastlio_rosbag','--input',str(inp),'--output',str(out)];main()
    ts=get_typestore(Stores.ROS1_NOETIC)
    with Reader(out) as reader:
        topics=[c.topic for c in reader.connections];assert '/fastlio/lidar' in topics and '/fastlio/imu' in topics
        assert sum(1 for _ in reader.messages())==2
