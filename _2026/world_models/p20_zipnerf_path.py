import json
import sys

import numpy as np


def to_zipnerf(c2w_cv, transform):
    transform = np.array(transform)
    scale = np.cbrt(np.linalg.det(transform[:3, :3]))
    c2w = np.array(c2w_cv, dtype=float)
    c2w[:3, 1:3] *= -1
    out = np.zeros((3, 4))
    out[:, :3] = transform[:3, :3] / scale @ c2w[:3, :3]
    out[:, 3] = transform[:3, :3] @ c2w[:3, 3] + transform[:3, 3]
    return out


if __name__ == '__main__':
    path_npz, calibration_path, out_path = sys.argv[1:4]
    d = np.load(path_npz)
    transform = json.load(open(calibration_path))['colmap_to_world_transform']
    poses = np.array([to_zipnerf(c, transform) for c in d['c2w']])
    np.save(out_path, poses)
    focal = d['height'] / 2 / np.tan(np.radians(d['fov_y_degrees']) / 2)
    print(f'{len(poses)} poses -> {out_path}, focal {focal:.3f}')
