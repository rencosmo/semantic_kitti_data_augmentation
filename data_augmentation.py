#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import numpy as np
# import matplotlib.pyplot as plt
import open3d as o3d
import os

def boundingbox(scan, label, bound):
  xmin = bound[0]
  xmax = bound[1]
  ymin = bound[2]
  ymax = bound[3]
  index = np.where( (scan[:, 0]>xmin) & (scan[:, 0]<xmax) & (scan[:, 1]>ymin) & (scan[:, 1]<ymax) )
  return scan[index[0], :], label[index[0]]

def cluster_translate(cluster, anchor, model_label):
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
  
  cluster_label = np.full((cluster.shape[0],), model_label)
 
  return cluster_trans, cluster_label


def injectObj(scan, label, model_path, model_label, bb_dim, model_num, freeSpaceTypes, freeSpaceTypesProb):
# Input:	
#   scan, label: original scan and label from semantic kitti 
#   model_path: path of model files
#   bb_dim: bounding box size which is used to find the free space for injection
#   model_num: how many models are injected into to background
#   freeSpaceTypes: list, lables for free space search. usually use road, parking, sidewalk, other ground [40, 44, 48, 49] 
#   freeSpaceTypesProb: list, possibility of the anchor point position. For example, freespaceTypes is [40, 44, 48, 49]
#                       freeSpaceTypesProb can be [0.7, 0.1, 0.1, 0.1]. It means that 70% of the anchor points are on the road,
#                       10% on each other three classes
# Output:
#   scan, Nx4 numpy array, point cloud with the injected 
#   label, labels after model injection\

  # Points number in the original point cloud
  ori_scan_num = scan.shape[0]

  # Read model library
  model_names = os.listdir(model_path)
  model_distances = []

  for model_name in model_names:
    begin = model_name.find('_')
    end = model_name.rfind('_')
    model_dis = model_name[begin+1:end]
    model_distances.append(float(model_dis)/100.0)
  model_distances = np.asarray(model_distances)

  # Find points of the free space
  freespace = np.empty([0, 4])
  freeSpaceTypesProbVec = np.empty([0])
  for i, freeSpaceType in enumerate(freeSpaceTypes):
    index = np.where(label==freeSpaceType)
    if index[0].shape[0]>0:
      freespace = np.vstack((freespace, scan[index[0], :]))
      prob_part = np.full(scan[index[0], :].shape[0], freeSpaceTypesProb[i])
      freeSpaceTypesProbVec = np.hstack((freeSpaceTypesProbVec, prob_part))

  # For uniform sampling base on the density
  freespace_dense = (freespace[:,0]**2 + freespace[:,1]**2)**(3/2)/(freespace[:,0]**2 + freespace[:,1]**2+freespace[:,2]**2)**(1/2) * freeSpaceTypesProbVec
  # freespace_dense = (freespace[:,0]**2 + freespace[:,1]**2) * freeSpaceTypesProbVec
  freespace_dense = freespace_dense/np.sum(freespace_dense)

  # find a freespace point randomly
  if freespace.shape[0] > 0:
    for i in range(0, model_num):
      ind = np.random.choice(np.arange(0,freespace.shape[0],1), 1, p=freespace_dense)
      bound = [freespace[ind, 0]-bb_dim/2, freespace[ind, 0]+bb_dim/2, freespace[ind, 1]-bb_dim/2, freespace[ind, 1]+bb_dim/2]
      box_points, box_label = boundingbox(scan, label, bound)
      dh = np.max(box_points[:, 2])-np.min(box_points[:, 2])
      anchor = [freespace[ind, 0][0], freespace[ind, 1][0], np.max(box_points[:, 2])]
      # no other object in the bounding box
      if dh<0.3:
        # distance of the bounding box
        distance = np.sqrt(freespace[ind, 0]*freespace[ind, 0] + freespace[ind, 1]*freespace[ind, 1])
        # find the model with suitable distance
        model_index = np.where( (model_distances<=distance) & (model_distances>=distance-10 ) )
        if model_index[0].shape[0]>0:
          model_num = np.random.choice(model_index[0], 1) # chosen a model
          cluster = np.load(model_path+model_names[model_num[0]]) # load model
          cluster_trans, cluster_label = cluster_translate(cluster, anchor, model_label)
          scan = np.vstack((scan, cluster_trans))
          label = np.hstack((label, cluster_label))

  scan, label = scanline_shadow_gen(scan, label, ori_scan_num)

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
  proj_W=2048
  proj_fov_up=3.0
  proj_fov_down=-25.0
  proj_fov_up = proj_fov_up / 180.0 * np.pi      # field of view up in rad
  proj_fov_down = proj_fov_down / 180.0 * np.pi       # field of view down in rad
  fov = abs(proj_fov_down) + abs(proj_fov_up)    # get field of view total in rad

  # Project  
  depth_ori = np.linalg.norm(scan_ori[:, :3], 2, axis=1)
  yaw_ori = -np.arctan2(scan_ori[:, 1], scan_ori[:, 0])
  pitch_ori = np.arcsin(scan_ori[:, 2] / depth_ori)

  proj_x_ori = 0.5 * (yaw_ori / np.pi + 1.0)                       # in [0.0, 1.0]
  proj_y_ori = 1.0 - (pitch_ori + abs(proj_fov_down)) / fov        # in [0.0, 1.0]

  proj_x_ori *= proj_W                              # in [0.0, W]
  proj_y_ori *= proj_H                              # in [0.0, H]

  proj_x_ori = np.floor(proj_x_ori)
  proj_x_ori = np.minimum(proj_W - 1, proj_x_ori)
  proj_x_ori = np.maximum(0, proj_x_ori).astype(np.int32)   # in [0,W-1]
  
  proj_y_ori = np.floor(proj_y_ori)
  proj_y_ori = np.minimum(proj_H - 1, proj_y_ori)
  proj_y_ori = np.maximum(0, proj_y_ori).astype(np.int32)   # in [0,H-1]

  # Project injected points into range image
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

  idx_inject_pts = proj_indices_inject[proj_indices_inject>0]
  scan_inject = scan_inject[idx_inject_pts, :]
  label_inject = label_inject[idx_inject_pts]

  proj_bool = (proj_indices_inject>0)
  
  ind_no_inject = np.where( proj_bool[proj_y_ori, proj_x_ori]==False )
  scan_aug = scan_ori[ind_no_inject[0], :]
  label_aug = label_ori[ind_no_inject[0]]

  scan_aug = np.vstack((scan_aug, scan_inject))
  label_aug = np.hstack((label_aug, label_inject))

  return scan_aug, label_aug

def scan_display(scan, label):
# Display the labelled point cloud of semantic kitti
  color_map = {0: [0, 0, 0],
  1: [0, 0, 255],
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 80, 100],
  15: [150, 60, 30],
  16: [255, 0, 0],
  18: [180, 30, 80],
  20: [255, 0, 0],
  30: [30, 30, 255],
  31: [200, 40, 255],
  32: [90, 30, 150],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [75, 0, 175],
  50: [0, 200, 255],
  51: [50, 120, 255],
  52: [0, 150, 255],
  60: [170, 255, 150],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [255, 255, 50],
  252: [245, 150, 100],
  256: [255, 0, 0],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  257: [250, 80, 100],
  258: [180, 30, 80],
  259: [255, 0, 0]}

  colors = []
  for key in label:
    colors.append( color_map.get(key) )

  colors = np.fliplr(np.array(colors))/255

  pcd = o3d.geometry.PointCloud()
  print(scan.shape, colors.shape)
  pcd.points = o3d.utility.Vector3dVector(scan[:, 0:3])
  pcd.colors = o3d.utility.Vector3dVector(colors)
  o3d.visualization.draw_geometries([pcd])
  

if __name__ == '__main__':

  # read point cloud
  filename_points = '/home/cosmo/workspace/Datasets/dataset/sequences/00/velodyne/000000.bin'
  print(filename_points)
  scan = np.fromfile(filename_points, dtype=np.float32)
  scan = scan.reshape((-1, 4))

  # read label
  filename_label = '/home/cosmo/workspace/Datasets/dataset/sequences/00/labels/000000.label'
  label = np.fromfile(filename_label, dtype=np.uint32)
  label = label.reshape((-1)) & 0xFFFF

  model_path = '/home/cosmo/workspace/dataaug/person/'

  ori_scan_num = scan.shape[0]
  bb_dim = 1.0
  model_label = 30
  scan, label = injectObj(scan, label, model_path, model_label, bb_dim, 100, [40, 44, 48], [0.80, 0.1, 0.1])
  scan_display(scan, label)