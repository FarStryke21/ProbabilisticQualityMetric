from utils_poseEstimation import *
import open3d as o3d
import numpy as np
import copy

target = loadPCD("data/chair_target.ply")
source = loadPCD("data/chair_source.ply")

voxel_size = 0.05

# draw_registration_result(source, target, np.identity(4))

# 1. Downsample the point clouds and get the FPFH features
source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

# Coarse registration
global_registration_result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

# Refine registration
refine_registration_result = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, global_registration_result.transformation)

x, y, z, roll, pitch, yaw = extract_translation_and_euler_angles(refine_registration_result.transformation)

print("Translation: {}, {}, {}".format(x, y, z))
print("Euler Angles: {}, {}, {}".format(roll, pitch, yaw))

# Find origins
source_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
target_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

# Transform the source origin
source_origin.transform(refine_registration_result.transformation)

# Visualize the point clouds and their origins
source_copy = copy.deepcopy(source) 
source_copy.transform(refine_registration_result.transformation)
o3d.visualization.draw_geometries([source_copy, target, source_origin, target_origin])



