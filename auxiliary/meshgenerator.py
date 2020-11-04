#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import open3d as o3d
import numpy as np
import numba as nb

import matplotlib.pyplot as plt

class meshGen:
  """Class that creates and handles a visualizer for a pointcloud"""

  def __init__(self, points):
    self.points = points
    self.upsample()
    dis_vec = np.sqrt(points[:, 0]*points[:, 0] + points[:, 1]*points[:, 1])
    self.dis_mean = np.average( dis_vec )

  # 2D grid interplation (does not work well)
  def upsample_2d(self):
    azi = np.arctan2(self.points[:, 1], self.points[:, 0])
    ele = np.arctan2(self.points[:, 2], np.sqrt(self.points[:, 0]*self.points[:, 0]+self.points[:, 1]*self.points[:, 1]) )
    sphere_coors = np.hstack( (azi.reshape(-1, 1), ele.reshape(-1, 1)) )

    grid_azi = np.arange(azi.min(), azi.max(), 0.0005)
    grid_ele = np.arange(ele.min(), ele.max(), 0.0005)

    xx, yy = np.meshgrid(grid_azi, grid_ele)
    xxx, yyy = xx.flatten(), yy.flatten()
    interp_sphere_coors = np.hstack( (xxx.reshape(-1, 1), yyy.reshape(-1, 1)) )

    from scipy.interpolate import griddata
    points_us = griddata(sphere_coors, self.points, interp_sphere_coors, method='linear')
    points_us = points_us[~np.isnan(points_us)]
    points_us = points_us.reshape(-1, 4)

    self.points_us = points_us

  # Upsampling the clusters
  #@nb.jit(???,nopython=True)
  def upsample(self):
    # increase the horinzontal density
    points = self.points
    for i in range(0, self.points.shape[0]-1):
      gap = np.linalg.norm(self.points[i+1, 0:3] - self.points[i, 0:3])
      if gap<0.05:
        points_add = (self.points[i+1, :] - self.points[i, :])/4
        for k in range(1, 4):
          interp_point = self.points[i, :] + points_add * k
          points = np.r_[points, interp_point.reshape(-1, 4)]

    azi_interval = 0.0006
    azi = np.arctan2(points[:, 1], points[:, 0])
    ele = np.arctan2(points[:, 2], np.sqrt(points[:, 0]*points[:, 0]+points[:, 1]*points[:, 1]) )
    azi_step = np.arange(azi.min(), azi.max(), azi_interval)
    
    points_us = np.empty([0, 4])
    for azi0 in azi_step:
      index = np.where( (azi>=azi0) & (azi<azi0+azi_interval) )
      col_points = points[index]
      col_ele = ele[index]
      col_index = np.argsort(col_ele)
      col_ele_sorted = col_ele[col_index]
      col_points_sorted = col_points[col_index]
      for j in range(0, col_points_sorted.shape[0]-1):
        gap_vert = np.linalg.norm(col_points_sorted[j+1, 0:3] - col_points_sorted[j, 0:3])
        if gap_vert < 0.2:
          points_add = (col_points_sorted[j+1, :] - col_points_sorted[j, :])/6
          for k in range(1, 6):
            interp_point = col_points_sorted[j, :] + points_add * k
            points_us = np.r_[points_us, interp_point.reshape(-1, 4)]
        else:
          points_us = np.r_[points_us, col_points_sorted[j+1, :].reshape(-1, 4)]
          points_us = np.r_[points_us, col_points_sorted[j, :].reshape(-1, 4)]

    self.points_us = points_us


  def tri_mesh(self):
    xyz = self.points[:, 0:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals()
    self.pcd = pcd
    depth = 6
    scale = 1.1
    width = 0
    linear_fit = False

    # alpha = 0.05
    # self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    # self.mesh, self.densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit)

    radii = [0.05, 0.05, 0.05, 0.05]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    self.mesh = rec_mesh

  def show_mesh(self):
    xyz = self.points_us[:, 0:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])

  def save_points_us(self, header):
    outfile = header+"_"+str( int( np.round(self.dis_mean*100) ) )+"_"+str(self.points_us.shape[0])+".npy"
    np.save(outfile, self.points_us)

