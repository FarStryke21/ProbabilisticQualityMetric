import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt
from math import *

def loadPCD(filename):
    pcd = o3d.io.read_point_cloud(filename) 
    return pcd

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    o3d.visualization.draw_geometries([source_temp, target_temp]) #,
                                    #   zoom=0.4459,
                                    #   front=[0.9288, -0.2951, -0.2242],
                                    #   lookat=[1.6784, 2.0612, 1.4451],
                                    #   up=[-0.3402, -0.9189, -0.1996])
    
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, initial_transformation):  
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)

    radius_normal = voxel_size * 2
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def project_pcd_to_depth(pcd, camera_pose, scale = (480, 480), depth_max=1000.0, depth_scale=1000.0, focal_dist=400.0):
    width, height = scale  
    pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd)
    intrinsics =o3d.core.Tensor([[focal_dist, 0     , width * 0.5], 
                                [0     , focal_dist, height * 0.5],
                                [0     , 0     , 1]])
    extrinsics = o3d.core.Tensor(camera_pose)
    depth_reproj = pcd_t.project_to_depth_image(width,
                                            height,
                                            intrinsics,
                                            extrinsics,
                                            depth_scale=depth_scale,
                                            depth_max=depth_max)
      
    depth_mat = np.asarray(depth_reproj.to_legacy())
    plot = plt.imshow(depth_mat)
    
    return depth_mat, plot

def get_projections(C1, O1, C2, O2, save = False):
    I11, plot_I11 = project_pcd_to_depth(C1, O1)
    I12, plot_I12 = project_pcd_to_depth(C1, O2)
    I21, plot_I21 = project_pcd_to_depth(C2, O1)
    I22, plot_I22 = project_pcd_to_depth(C2, O2)

    if save:
        plt.imsave('I11.png', I11)
        plt.imsave('I12.png', I12)
        plt.imsave('I21.png', I21)
        plt.imsave('I22.png', I22)

    return I11, I12, I21, I22

def get_p(I11, I12, I21, I22, sigma = 0.01):
    def CDF(x):
        return 1/2 * (1 + erf(x/sqrt(2)))

    def p_calc(delJ, sigma):
        p = 1 - (CDF(delJ/sigma) - CDF(-delJ/sigma))
        return p
    
    I_pair = [(I11, I21),(I22, I12)]
    # I_pair = [(I11, I12),(I22, I21)]
    net_p = []

    for i,pair in enumerate(I_pair):
        I1 = pair[0].flatten()
        I2 = pair[1].flatten()
        p = np.zeros_like(I1)
        for i in range(I1.shape[0]):
            if I1[i] != 0 and I2[i] != 0:
                p[i] = p_calc(I1[i] - I2[i], sigma)
            elif I1[i] == 0 and I2[i] == 0:
                p[i] = 0
            elif I1[i] == 0 and I2[i] != 0:
                p[i] = p_calc(2*sigma, sigma)

        net_p.extend(p.tolist())

    net_p = np.array(net_p)
    net_p = net_p.flatten()
    M = np.count_nonzero(net_p)
    match_p = np.sum(net_p)/M

    return match_p

def extract_translation_and_euler_angles(transformation_matrix):
    """Extract translation and Euler angles from a transformation matrix."""
    x, y, z = transformation_matrix[:3, 3]

    # Extract rotation matrix
    rotation_matrix = transformation_matrix[:3, :3]

    # Roll (rotation around x-axis)
    roll = atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    
    # Pitch (rotation around y-axis)
    pitch = atan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
    
    # Yaw (rotation around z-axis)
    yaw = atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Convert angles from radians to degrees
    roll_deg = degrees(roll)
    pitch_deg = degrees(pitch)
    yaw_deg = degrees(yaw)

    return x, y, z, roll_deg, pitch_deg, yaw_deg