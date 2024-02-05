from quality_utils import *
import open3d as o3d
import numpy as np
import copy

target = loadPCD("data/chair_target.ply")
source = loadPCD("data/chair_source.ply")

voxel_size = 0.05

# 1. Downsample the point clouds and get the FPFH features
source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

# Coarse registration
global_registration_result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

# Refine registration
refine_registration_result = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, global_registration_result.transformation)

# Find origins
source_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
target_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

# Transform the source origin
source_origin.transform(refine_registration_result.transformation)

# Visualize the point clouds and their origins
source_copy = copy.deepcopy(source) 
source_copy.transform(refine_registration_result.transformation)

# Uncomment below to introduce translation error
# source_copy.translate([0.25, 0, 0])

# Uncomment below to introduce rotation error (rx, ry, rz)
# source_copy.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.radians(np.asarray((45, 0, 0)))))

# o3d.visualization.draw_geometries([source_copy, target, source_origin, target_origin])

T = refine_registration_result.transformation
I11, I12, I21, I22 = get_projections(target, np.identity(4), source_copy, np.linalg.inv(T), save=True)

p = get_p(I11, I12, I21, I22, sigma=0.01)
print("Alignment Quality: ", p)