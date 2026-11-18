import sys
from pathlib import Path

import numpy as np


def quat_to_rot(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def read_images(path):
    names, centers, rots = [], [], []
    lines = [l for l in path.read_text().splitlines() if l and not l.startswith('#')]
    for line in lines[::2]:
        parts = line.split()
        q = [float(v) for v in parts[1:5]]
        t = np.array([float(v) for v in parts[5:8]])
        rot = quat_to_rot(*q)
        names.append(parts[9])
        centers.append(-rot.T @ t)
        rots.append(rot)
    order = np.argsort(names)
    return ([names[i] for i in order], np.array(centers)[order], np.array(rots)[order])


def read_points(path, min_track=3, max_error=1.5):
    xyz, rgb = [], []
    for line in path.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if (len(parts) - 8) // 2 < min_track or float(parts[7]) > max_error:
            continue
        xyz.append([float(v) for v in parts[1:4]])
        rgb.append([int(v) for v in parts[4:7]])
    return np.array(xyz), np.array(rgb, dtype=np.uint8)


def rotation_to_z(up):
    up = up / np.linalg.norm(up)
    axis = np.cross(up, [0, 0, 1])
    s, c = np.linalg.norm(axis), up[2]
    if s < 1e-9:
        return np.eye(3)
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def main(model_txt_dir, out_path, radius=4.0):
    model = Path(model_txt_dir)
    names, centers, rots = read_images(model / 'images.txt')
    xyz, rgb = read_points(model / 'points3D.txt')

    align = rotation_to_z(-rots[:, 1, :].mean(axis=0))
    centers = centers @ align.T
    xyz = xyz @ align.T
    right = rots[:, 0, :] @ align.T
    up = -rots[:, 1, :] @ align.T
    forward = rots[:, 2, :] @ align.T

    origin = np.median(centers, axis=0)
    origin[2] = np.percentile(xyz[:, 2], 2)
    scale = radius / np.percentile(np.linalg.norm((centers - origin)[:, :2], axis=1), 90)
    centers = (centers - origin) * scale
    xyz = (xyz - origin) * scale

    lo, hi = np.percentile(xyz, [1, 99], axis=0)
    keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    np.savez_compressed(out_path, names=np.array(names), centers=centers, right=right,
                        up=up, forward=forward, points=xyz[keep], colors=rgb[keep],
                        align=align, origin=origin, scale=scale)
    print(f'{len(names)} cameras, {keep.sum()} points -> {out_path}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
