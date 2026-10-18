"""Lightweight scan-to-scan fallback for synthetic FAST-LIO observations.

This is only activated when FAST-LIO has no effective scan-to-map points. It
uses the same FAST-LIO LiDAR frame and returns body poses with the configured
LiDAR-to-IMU translation, so the OneMap evaluation can distinguish optimizer
failure from lack of geometric overlap.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def rigid(a, b):
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a-ca).T @ (b-cb))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = cb - r @ ca
    return r, t


def downsample(p, voxel=0.08):
    if len(p) == 0: return p
    key = np.floor(p / voxel).astype(np.int32)
    _, ix = np.unique(key, axis=0, return_index=True)
    return p[np.sort(ix)]


def register(prev, cur, max_dist=0.45, iterations=8):
    # Estimate T_prev_cur: prev ~= R * cur + t.
    src, dst = downsample(cur), downsample(prev)
    rtot, ttot = np.eye(3), np.zeros(3)
    tree = cKDTree(dst)
    for _ in range(iterations):
        moved = src @ rtot.T + ttot
        d, ix = tree.query(moved, k=1, workers=-1)
        keep = np.isfinite(d) & (d < max_dist)
        if keep.sum() < 30: break
        r, t = rigid(moved[keep], dst[ix[keep]])
        rtot, ttot = r @ rtot, r @ ttot + t
        if np.linalg.norm(t) < 1e-4 and np.linalg.norm(r - np.eye(3)) < 1e-4: break
    return rtot, ttot, int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--lidar-to-body', nargs=3, type=float, default=[.3, 0., .3])
    args = ap.parse_args()
    files = sorted((args.input/'lidar').glob('*.npz'))
    scans = [np.asarray(np.load(f)['points_body'], np.float64) for f in files]
    poses = []
    t_world_lidar = np.eye(4)
    quality = []
    for i, scan in enumerate(scans):
        if i:
            r, t, n = register(scans[i-1], scan)
            step = np.eye(4); step[:3,:3] = r; step[:3,3] = t
            # ``step`` maps current-scan points into the previous scan frame:
            # p_prev = step * p_cur.  Therefore the current world pose is
            # T_world_prev * step (not its inverse).
            t_world_lidar = t_world_lidar @ step
            quality.append(n)
        else:
            quality.append(len(scan))
        body = t_world_lidar.copy(); body[:3,3] += body[:3,:3] @ np.asarray(args.lidar_to_body)
        from scipy.spatial.transform import Rotation
        poses.append({'index': i, 'timestamp': i*.1, 'position': body[:3,3].tolist(), 'rotation_xyzw': Rotation.from_matrix(body[:3,:3]).as_quat().tolist(), 'split':'sensor', 'source':'scanmatch_fallback'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'poses':poses, 'registration_inliers':quality, 'source':'scanmatch_fallback', 'lidar_to_body':args.lidar_to_body}, indent=2)+'\n')
    print(json.dumps({'frames':len(poses), 'median_inliers':float(np.median(quality)), 'output':str(args.output)}))


if __name__ == '__main__': main()
