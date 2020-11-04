#!/usr/bin/env python3
import numpy as np
import open3d as o3d
import sys, os

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
    key = input('Delete? (input d): ')
    if key=='d':
      print('Deleted '+model_path+model_name)
      os.remove(model_path+model_name)
    else:
      print('Keep '+model_path+model_name)
