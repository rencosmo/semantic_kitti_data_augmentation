#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os

def boundingbox(scan, label, bound):
  xmin = bound[0]
  xmax = bound[1]
  ymin = bound[2]
  ymax = bound[3]
  index = np.where( (scan[:, 0]>xmin) & (scan[:, 0]<xmax) & (scan[:, 1]>ymin) & (scan[:, 1]<ymax) )
  return scan[index[0], :], label[index[0]]

def cluster_translate(cluster, anchor):
  cluster_x_mean = np.average(cluster[:, 0])
  cluster_y_mean = np.average(cluster[:, 1])
  cluster_z_min = np.min(cluster[:, 2])
  cluster[:, 0] = cluster[:, 0] - cluster_x_mean
  cluster[:, 1] = cluster[:, 1] - cluster_y_mean
  cluster[:, 2] = cluster[:, 2] - cluster_z_min

  # random scale gain
  vertical_gain = np.random.uniform(0.9, 1.1)
  horizontal_gain = np.random.uniform(0.9, 1.1)
  cluster[:, 0] = horizontal_gain * cluster[:, 0]
  cluster[:, 1] = horizontal_gain * cluster[:, 1]
  cluster[:, 2] = vertical_gain * cluster[:, 2]

  # compute the rotation angle
  a = np.array([cluster_x_mean, cluster_y_mean])
  b = np.array([anchor[0], anchor[1]])
  rot_angle = -np.arccos( a.dot(b)/(np.linalg.norm(a) * np.linalg.norm(b)) ) * np.sign(np.cross(a, b))
  
  # rotate the model
  cluster_trans = np.empty_like(cluster)
  cluster_trans[:, 0] = np.cos(rot_angle)*cluster[:, 0] + np.sin(rot_angle)*cluster[:, 1]
  cluster_trans[:, 1] = -np.sin(rot_angle)*cluster[:, 0] + np.cos(rot_angle)*cluster[:, 1]
  cluster_trans[:, 2:] = cluster[:, 2:]

  # translate the cluster to anchor point
  cluster_trans[:, 0] = cluster_trans[:, 0] + anchor[0]
  cluster_trans[:, 1] = cluster_trans[:, 1] + anchor[1]
  cluster_trans[:, 2] = cluster_trans[:, 2] + anchor[2]
  
  cluster_label = np.full((cluster.shape[0],), 30)
 
  return cluster_trans, cluster_label


def injectObj(scan, label):
  # Read model library
  model_path = '/home/cosmo/workspace/dataaug/person/'
  model_names = os.listdir(model_path)
  model_distances = []

  for model_name in model_names:
    begin = model_name.find('_')
    end = model_name.rfind('_')
    model_dis = model_name[begin+1:end]
    model_distances.append(float(model_dis)/100.0)
  model_distances = np.asarray(model_distances)

  # Find the road, sidewalk and parking in the original scan  
  index0 = np.where(label==48)
  sidewalk = scan[index0[0], :]
  
  index1 = np.where(label==40)
  road = scan[index1[0], :]

  index2 = np.where(label==44)
  parking = scan[index2[0], :]

  # find a sidewalk point randomly
  for i in range(0, 800):
    ind = np.random.choice(sidewalk.shape[0]-1, 1)
    bound = [sidewalk[ind, 0]-1, sidewalk[ind, 0]+1, sidewalk[ind, 1]-1, sidewalk[ind, 1]+1]
    box_points, box_label = boundingbox(scan, label, bound)
    dh = np.max(box_points[:, 2])-np.min(box_points[:, 2])
    anchor = [sidewalk[ind, 0][0], sidewalk[ind, 1][0], np.max(box_points[:, 2])]
    # no other object in the bounding box
    if dh<0.3:
      # distance of the bounding box
      distance = np.sqrt(sidewalk[ind, 0]*sidewalk[ind, 0] + sidewalk[ind, 1]*sidewalk[ind, 1])
      # find the model with suitable distance
      model_index = np.where( (model_distances<10.0) & (model_distances>3.0) ) #distance)
      if model_index[0].shape[0]>0:
        model_num = np.random.choice(model_index[0], 1) # chosen a model
        cluster = np.load(model_path+model_names[model_num[0]]) # load model
        cluster_trans, cluster_label = cluster_translate(cluster, anchor)
        scan = np.vstack((scan, cluster_trans))
        label = np.hstack((label, cluster_label))
  return scan, label

def scanline_shadow_gen(scan, label, ori_scan_num):
  # orignal points
  scan_ori = scan[:ori_scan_num, :]
  label_ori = label[:ori_scan_num]
  # injected points
  scan_inject = scan[ori_scan_num:, :]
  label_inject = label[ori_scan_num:]

  # generate the range image of original points
  proj_H=64
  proj_W=1024
  proj_fov_up=3.0
  proj_fov_down=-25.0
  proj_fov_up = proj_fov_up / 180.0 * np.pi      # field of view up in rad
  proj_fov_down = proj_fov_down / 180.0 * np.pi       # field of view down in rad
  fov = abs(proj_fov_down) + abs(proj_fov_up)    # get field of view total in rad

  '''
  depth_ori = np.linalg.norm(scan_ori[:, :3], 2, axis=1)
  yaw_ori = -np.arctan2(scan_ori[:, 1], scan_ori[:, 0])
  pitch_ori = np.arcsin(scan_ori[:, 2] / depth_ori)

  # get projections in image coords
  proj_x_ori = 0.5 * (yaw_ori / np.pi + 1.0)                  # in [0.0, 1.0]
  proj_y_ori = 1.0 - (pitch_ori + abs(fov_down)) / fov        # in [0.0, 1.0]

  proj_x_ori *= proj_W                              # in [0.0, W]
  proj_y_ori *= proj_H                              # in [0.0, H]

  proj_x_ori = np.floor(proj_x_ori)
  proj_x_ori = np.minimum(proj_W - 1, proj_x_ori)
  proj_x_ori = np.maximum(0, proj_x_ori).astype(np.int32)   # in [0,W-1]
  
  proj_y_ori = np.floor(proj_y_ori)
  proj_y_ori = np.minimum(proj_H - 1, proj_y_ori)
  proj_y_ori = np.maximum(0, proj_y_ori).astype(np.int32)   # in [0,H-1]
  '''

  # Project injected points 
  depth_inject = np.linalg.norm(scan_inject[:, :3], 2, axis=1)
  yaw_inject = -np.arctan2(scan_inject[:, 1], scan_inject[:, 0])
  pitch_inject = np.arcsin(scan_inject[:, 2] / depth_inject)

  proj_x_inject = 0.5 * (yaw_inject / np.pi + 1.0)                  # in [0.0, 1.0]
  proj_y_inject = 1.0 - (pitch_inject + abs(proj_fov_down)) / fov        # in [0.0, 1.0]

  proj_x_inject *= proj_W                              # in [0.0, W]
  proj_y_inject *= proj_H                              # in [0.0, H]

  proj_x_inject = np.floor(proj_x_inject)
  proj_x_inject = np.minimum(proj_W - 1, proj_x_inject)
  proj_x_inject = np.maximum(0, proj_x_inject).astype(np.int32)   # in [0,W-1]
  
  proj_y_inject = np.floor(proj_y_inject)
  proj_y_inject = np.minimum(proj_H - 1, proj_y_inject)
  proj_y_inject = np.maximum(0, proj_y_inject).astype(np.int32)   # in [0,H-1]
  
  indices_inject = np.arange(depth_inject.shape[0])
  order_inject= np.argsort(depth_inject)[::-1]
  depth_inject = depth_inject[order_inject]
  indices_inject = indices_inject[order_inject]
  proj_y_inject = proj_y_inject[order_inject]
  proj_x_inject = proj_x_inject[order_inject]

  proj_indices_inject = np.full((proj_H, proj_W), -1,
                                 dtype=np.int32)
  proj_indices_inject[proj_y_inject, proj_x_inject] = indices_inject

  plt.matshow(proj_indices_inject)
  plt.show()
  idx_inject_pts = proj_indices_inject[proj_indices_inject>0]
  scan_inject = scan_inject[idx_inject_pts, :]
  label_inject = label_inject[idx_inject_pts]

  scan = np.vstack((scan_ori, scan_inject))
  label = np.hstack((label_ori, label_inject))

  return scan, label



def scan_display(scan, label):
  '''
  road_pcd = o3d.geometry.PointCloud()
  road_pcd.points = o3d.utility.Vector3dVector(road[:, 0:3])
  road_pcd.paint_uniform_color([1, 0, 0])
  '''

  sidewalk_ind = np.where(label==48)
  sidewalk_pcd = o3d.geometry.PointCloud()
  sidewalk_pcd.points = o3d.utility.Vector3dVector(scan[sidewalk_ind[0], 0:3])
  sidewalk_pcd.paint_uniform_color([0.294, 0, 0.294])

  background_ind = np.where((label!=30) & (label!=48))
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(scan[background_ind[0], 0:3])
  pcd.paint_uniform_color([0.65, 0.65, 0.65])

  person_ind = np.where(label==30)
  person_pcd = o3d.geometry.PointCloud()
  person_pcd.points = o3d.utility.Vector3dVector(scan[person_ind[0], 0:3])
  person_pcd.paint_uniform_color([1, 0, 0]) 

  o3d.visualization.draw_geometries([pcd, person_pcd, sidewalk_pcd])
  

if __name__ == '__main__':

  # read point cloud
  filename_points = '/home/cosmo/workspace/Datasets/dataset/sequences/00/velodyne/000000.bin'
  print(filename_points)
  scan = np.fromfile(filename_points, dtype=np.float32)
  scan = scan.reshape((-1, 4))

  # read label
  filename_label = '/home/cosmo/workspace/Datasets/dataset/sequences/00/labels/000000.label'
  label = np.fromfile(filename_label, dtype=np.uint32)
  label = label.reshape((-1))

  ori_scan_num = scan.shape[0]
  scan, label = injectObj(scan, label)

  scan, label = scanline_shadow_gen(scan, label, ori_scan_num)
  
  scan_display(scan, label)





