## Probabilistic Quality Metric for registered Point Clouds of Objects


### Support for Python2

Open3d 0.9.0 supports Python 2.7
The implementation uptil registration is function with Python2. Full implementation using Python2 is in progress. use **poseEstimation.py** to extract the pose difference between the two pointclouds

--------
### Full Support on the main branch

This repository is a modified implementation of **Analyzing the Quality of Matched 3D Point Clouds of Objects** by Igor Bogoslavskyi and Cyrill Stachniss. ([link](https://doi.org/10.1109/IROS.2017.8206584))

While the paper was developed for a dynamic real world set-up, this implementation is fine tuned for a lab environment and prioritizes accuracy over generalizability. The classification features mentioned in the paper have not been implemented. The idea here is to create a universal metric to compare quality of alignment in registered point clouds.

### Usage
The implementation is located in **pipeline.py**, with support functions provided in **quality_utils.py**. This repository is still under development is prone to updates in the future. 

The file **pipeline.py** takes in as input two unregistered point clouds and performs a two step alignment process. The initial global alignment makes use of RANSAC and FPFH features to coarsely align a down-sampled version of the point cloud, while fine alignment makes use of point to plane ICP algorithm to get a proper alignment.

This alignment is performed prior to calculating the quality metric. The probabilistic quality metric is a function of the several parameters used in developing the camera intrinsic and extrinsic parameters defined in the project_pcd_to_depth function in the utilities. Feel free to play around with them when tuning the metric with a ground truth. A value of around 0.8 to 0.9 is ideal. Anything less than 0.6 requires more tuning while anything above 1.0 indicates a need to relax the parameters.

The **metric_development.ipynb** notebook is a development playground and can be entirely ignored. Place all data files (point louds, meshes, etc.) in the data folder so that they can be references relatively without error.

### Requirements
The implementation has been developed and tested on a Windows system running Python 3.8. Open3d 0.18.0 is the desirable version required. Documentation for Open3d can be found [here](https://www.open3d.org/docs/release/index.html)

### References
1. I. Bogoslavskyi and C. Stachniss, "Analyzing the quality of matched 3D point clouds of objects," 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Vancouver, BC, Canada, 2017, pp. 6685-6690, https://doi.org/10.1109/IROS.2017.8206584.
2. Zhou, Q.-Y., Park, J., & Koltun, V. (2018). Open3D: A Modern Library for 3D Data Processing (Version 1). arXiv. https://doi.org/10.48550/ARXIV.1801.09847

