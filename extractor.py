#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import numpy as np
from auxiliary.laserscan import LaserScan, SemLaserScan
from auxiliary.laserscanextract import LaserScanExtract
from auxiliary.meshgenerator import meshGen

import matplotlib.pyplot as plt


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./visualize.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to visualize. No Default',
  )
  parser.add_argument(
      '--config', '-c',
      type=str,
      required=False,
      default="config/semantic-kitti.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )
  parser.add_argument(
      '--sequence', '-s',
      type=str,
      default="00",
      required=False,
      help='Sequence to visualize. Defaults to %(default)s',
  )
  parser.add_argument(
      '--predictions', '-p',
      type=str,
      default=None,
      required=False,
      help='Alternate location for labels, to use predictions folder. '
      'Must point to directory containing the predictions in the proper format '
      ' (see readme)'
      'Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_semantics', '-i',
      dest='ignore_semantics',
      default=False,
      action='store_true',
      help='Ignore semantics. Visualizes uncolored pointclouds.'
      'Defaults to %(default)s',
  )
  parser.add_argument(
      '--do_instances', '-di',
      dest='do_instances',
      default=False,
      action='store_true',
      help='Visualize instances too. Defaults to %(default)s',
  )
  parser.add_argument(
      '--offset',
      type=int,
      default=0,
      required=False,
      help='Sequence to start. Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_safety',
      dest='ignore_safety',
      default=False,
      action='store_true',
      help='Normally you want the number of labels and ptcls to be the same,'
      ', but if you are not done inferring this is not the case, so this disables'
      ' that safety.'
      'Defaults to %(default)s',
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Dataset", FLAGS.dataset)
  print("Config", FLAGS.config)
  print("Sequence", FLAGS.sequence)
  print("Predictions", FLAGS.predictions)
  print("ignore_semantics", FLAGS.ignore_semantics)
  print("do_instances", FLAGS.do_instances)
  print("ignore_safety", FLAGS.ignore_safety)
  print("offset", FLAGS.offset)
  print("*" * 80)

  # open config file
  try:
    print("Opening config file %s" % FLAGS.config)
    CFG = yaml.safe_load(open(FLAGS.config, 'r'))
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()

  # fix sequence name
  FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))

  # does sequence folder exist?
  scan_paths = os.path.join(FLAGS.dataset, "sequences",
                            FLAGS.sequence, "velodyne")
  if os.path.isdir(scan_paths):
    print("Sequence folder exists! Using sequence from %s" % scan_paths)
  else:
    print("Sequence folder doesn't exist! Exiting...")
    quit()

  # populate the pointclouds
  scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
      os.path.expanduser(scan_paths)) for f in fn]
  scan_names.sort()

  # does sequence folder exist?
  if not FLAGS.ignore_semantics:
    if FLAGS.predictions is not None:
      label_paths = os.path.join(FLAGS.predictions, "sequences",
                                 FLAGS.sequence, "predictions")
    else:
      label_paths = os.path.join(FLAGS.dataset, "sequences",
                                 FLAGS.sequence, "labels")
    if os.path.isdir(label_paths):
      print("Labels folder exists! Using labels from %s" % label_paths)
    else:
      print("Labels folder doesn't exist! Exiting...")
      quit()
    # populate the pointclouds
    label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(label_paths)) for f in fn]
    label_names.sort()

    # check that there are same amount of labels and scans
    if not FLAGS.ignore_safety:
      assert(len(label_names) == len(scan_names))

  # create a scan
  if FLAGS.ignore_semantics:
    scan = LaserScan(project=True)  # project all opened scans to spheric proj
  else:
    color_dict = CFG["color_map"]
    nclasses = len(color_dict)
    scan = SemLaserScan(nclasses, color_dict, project=True)

  # create a visualizer
  semantics = not FLAGS.ignore_semantics
  instances = FLAGS.do_instances
  if not semantics:
    label_names = None

  libclass = [11, 13, 15, 16, 18, 20, 30, 31, 32, 253, 254, 255, 256, 257, 258, 259]

  classes = {11: ["bicycle", 200, 2.5, 2.5, 1.0, 1.6],
             13: ["bus", 200, 5.0, 5.0, 2.0, 4.0],                                       #
             15: ["motorcycle", 200, 2.5, 2.5, 1.0, 1.6],
             16: ["on-rails", 200, 5.0, 5.0, 2.0, 4.0],                                  #
             18: ["truck", 200, 5.0, 5.0, 1.8, 4.0],
             20: ["other-vehicle", 200, 4.0, 4.0, 1.2, 2.6],
             30: ["person", 200, 1.8, 1.8, 1.2, 1.8],
             31: ["bicyclist", 200, 2.5, 2.5, 1.4, 1.8],
             32: ["motorcyclist", 200, 2.8, 2.8, 1.4, 1.8],                              #
             71: ["trunk", 80, 2.0, 2.0, 1.8, 4.5],
             80: ["pole", 80, 1.0, 1.0, 2.0, 4.5],
             81: ["traffic-sign", 30, 1.5, 1.5, 0.1, 2.0],
             99: ["other-object", 30, 3.0, 3.0, 0.8, 4.0],
             253: ["moving-bicyclist", 200, 2.5, 2.5, 1.4, 1.8],
             254: ["moving-person", 200, 1.8, 1.8, 1.2, 2.0],
             255: ["moving-motorcyclist", 200, 2.8, 2.8, 1.4, 1.8],
             256: ["moving-on-rails", 200, 5.0, 5.0, 2.0, 4.0],                          #
             257: ["moving-bus", 200, 5.0, 5.0, 2.0, 4.0],
             258: ["moving-truck", 200, 5.0, 5.0, 2.0, 4.0],
             259: ["moving-other-vehicle", 200, 5.0, 5.0, 1.2, 4.0]}

  for key in classes.keys():
    if key not in libclass:
      continue
    # Extract objects from LiDAR scans
    extractor = LaserScanExtract(scan=scan,
                       scan_names=scan_names,
                       label_names=label_names,
                       offset=FLAGS.offset,
                       semantics=semantics, instances=instances and semantics,
                       classid=key, min_point_num=classes[key][1])
    clusters = extractor.update_scan()

    # check the dimension of each cluster, if it is too big, there are more than one objects in the cluster
    for num, cluster in enumerate(clusters):
      cluster_dimensions = cluster.max(axis=0)-cluster.min(axis=0)

      if cluster_dimensions[0] < classes[key][2] and cluster_dimensions[1]<classes[key][3] and cluster_dimensions[2]>classes[key][4] and cluster_dimensions[2]<classes[key][5]:
        mesh = meshGen(cluster)
        # mesh.show_mesh()
        mesh.save_points_us(classes[key][0])