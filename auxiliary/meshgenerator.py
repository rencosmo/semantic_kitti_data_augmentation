#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import open3d as o3d
import numpy as np

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
  def upsample(self):
    azi_interval = 0.0032
    azi = np.arctan2(self.points[:, 1], self.points[:, 0])
    ele = np.arctan2(self.points[:, 2], np.sqrt(self.points[:, 0]*self.points[:, 0]+self.points[:, 1]*self.points[:, 1]) )
    azi_step = np.arange(azi.min(), azi.max(), azi_interval)
    
    points_us = np.empty([0, 4])
    for azi0 in azi_step:
      index = np.where( (azi>=azi0) & (azi<azi0+azi_interval) )
      col_points = self.points[index]
      col_ele = ele[index]
      col_index = np.argsort(col_ele)
      col_ele_sorted = col_ele[col_index]
      col_points_sorted = col_points[col_index]
      if col_ele_sorted.shape[0]>=4:
        ele_step = np.arange(col_ele_sorted.min(), col_ele_sorted.max(), azi_interval)
        x = np.interp(ele_step, col_ele_sorted, col_points_sorted[:, 0])
        y = np.interp(ele_step, col_ele_sorted, col_points_sorted[:, 1])
        z = np.interp(ele_step, col_ele_sorted, col_points_sorted[:, 2])
        i = np.interp(ele_step, col_ele_sorted, col_points_sorted[:, 3])
        col_group = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1), i.reshape(-1, 1)))
        points_us = np.append(points_us, col_group, axis=0)
        points_us = np.append(points_us,  col_points_sorted, axis=0)
      else:
        points_us = np.append(points_us,  col_points_sorted, axis=0)
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
    # print("============================: ", self.points_us.shape, self.points.shape)
    xyz = self.points_us[:, 0:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])

  def save_points_us(self, header):
    outfile = header+"_"+str( int( np.round(self.dis_mean*100) ) )+"_"+str(self.points_us.shape[0])+".npy"
    np.save(outfile, self.points_us)

