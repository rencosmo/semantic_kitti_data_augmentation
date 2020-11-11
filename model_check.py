#!/usr/bin/env python3
import numpy as np
import open3d as o3d
import sys, os

class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()

class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()

if __name__ == '__main__':
    model_path = sys.argv[1]
    print('Open '+model_path+'.')
    model_names = os.listdir(model_path)
    for model_name in model_names:
        print('Loading'+model_path+model_name)
        cluster = np.load(model_path+model_name)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster[:, 0:3])
        o3d.visualization.draw_geometries([pcd])
        getch = _Getch()
        key = getch()
        if key=='d':
            print('Deleted '+model_path+model_name)
            os.remove(model_path+model_name)
        else:
            print('Keep '+model_path+model_name)
