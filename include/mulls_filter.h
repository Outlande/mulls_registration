//
// This file is used for the filtering and feature point extraction of Point Cloud.
// Dependent 3rd Libs: PCL (>1.7)
// By Yue Pan
//

#ifndef _INCLUDE_MULLS_FILTER
#define _INCLUDE_MULLS_FILTER

//pcl
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/normal_space.h>
#include <pcl/filters/random_sample.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>


#include <vector>
#include <iostream>
#include <cfloat>

#include "mulls_util.h"
#include "pca.h"
#include <glog/logging.h>
#include <chrono>
#include <limits>
#include <time.h>

namespace mulls
{
class MullsFilter
{
  public:
	bool voxel_downsample(const pcl::PointCloud<MullsPoint>::Ptr &cloud_in, 
	                      pcl::PointCloud<MullsPoint>::Ptr &cloud_out, float voxel_size);

	bool xy_normal_balanced_downsample(pcl::PointCloud<MullsPoint>::Ptr &cloud_in_out,
									   int keep_number_per_sector, int sector_num);

	//fixed number random downsampling
	//when keep_number == 0, the output point cloud would be empty (in other words, the input point cloud would be cleared)
	bool random_downsample_pcl(pcl::PointCloud<MullsPoint>::Ptr &cloud_in_out, int keep_number);


	//fixed number random downsampling
	//when keep_number == 0, the output point cloud would be empty
	bool random_downsample_pcl(pcl::PointCloud<MullsPoint>::Ptr &cloud_in,
							pcl::PointCloud<MullsPoint>::Ptr &cloud_out, int keep_number);

	bool random_downsample(const pcl::PointCloud<MullsPoint>::Ptr &cloud_in,
						   pcl::PointCloud<MullsPoint>::Ptr &cloud_out, int downsample_ratio);

	bool random_downsample(pcl::PointCloud<MullsPoint>::Ptr &cloud_in_out, int downsample_ratio);

	//filter the ghost points under ground and the points on the ego-vehicle
	//ghost radius should be a bit larger than self radius
	//we assume that most of the ghost points are underground outliers within a radius from the laser scanner
	bool scanner_filter(const pcl::PointCloud<MullsPoint>::Ptr &cloud_in_out, float self_radius, 
	                    float ghost_radius, float z_min_thre_ghost, float z_min_thre_global);

	bool bbx_filter(const typename pcl::PointCloud<MullsPoint>::Ptr &cloud_in,
					typename pcl::PointCloud<MullsPoint>::Ptr &cloud_out, Bounds &bbx);

	bool bbx_filter(const typename pcl::PointCloud<MullsPoint>::Ptr &cloud_in_out, 
	                Bounds &bbx, bool delete_box = false);

	//extract stable points and then encode point cloud neighborhood feature descriptor (ncc: neighborhood category context) at the same time
	bool encode_stable_points(const typename pcl::PointCloud<MullsPoint>::Ptr &cloud_in,
							  typename pcl::PointCloud<MullsPoint>::Ptr &cloud_out,
							  const std::vector<pca_feature_t> &features,
							  const std::vector<int> &index_with_feature,
							  float min_curvature = 0.0,
							  int min_feature_point_num_neighborhood = 4,
							  int min_point_num_neighborhood = 8);

	//according to curvature
	bool non_max_suppress(pcl::PointCloud<MullsPoint>::Ptr &cloud_in_out, float non_max_radius,
						  bool kd_tree_already_built = false, const pcl::search::KdTree<MullsPoint>::Ptr &built_tree = NULL);

	//Brief: Use NMS to select those key points having locally maximal curvature
	//according to curvature
	bool non_max_suppress(pcl::PointCloud<MullsPoint>::Ptr &cloud_in,
						  pcl::PointCloud<MullsPoint>::Ptr &cloud_out, float nms_radius,
						  bool distance_adaptive_on = false, float unit_dist = 35.0,
						  bool kd_tree_already_built = false, const typename pcl::search::KdTree<MullsPoint>::Ptr &built_tree = NULL);

	// -------------------------------------------------------------------------------------------------------------------//
	// Two threshold Fast Ground filter
	// 1.Construct 2D grid
	// 2.Calculate the Minimum Z value in each grid
	// 3.For each grid, if its 8 neighbor grids' Minimum Z value is less than current grid's Minimum Z minus threshold1, then all the points in current grid would be seen as unground points
	// 4.Or, points whose Z value is larger than grid's Minimum Z plus threshold2 would be regarded as unground points. The rest points in the grid would be ground points.
	// (Estimate Ground Points' normal at the same time)
	// for multiple scan line mobile scanning data, we can detect the curb points at the same time according to ring information
	// ground_random_down_rate: 1 ground point in [ground_random_down_rate] ground points of each grid would be kept (for example, 20)
	// ground_random_down_down_rate: 1 ground point in [ground_random_down_down_rate] ground points of the already downsampled ground points would be kept (for example, 2)
	// Reference paper: Two-step adaptive extraction method for ground points and breaklines from lidar point clouds, ISPRS-J, Yang.B, Huang.R, et al.
	// -------------------------------------------------------------------------------------------------------------------//
	//current intensity_thre is for kitti dataset (TODO: disable it) 
	bool fast_ground_filter(const typename pcl::PointCloud<MullsPoint>::Ptr &cloud_in,
							typename pcl::PointCloud<MullsPoint>::Ptr &cloud_ground,
							typename pcl::PointCloud<MullsPoint>::Ptr &cloud_ground_down,
							typename pcl::PointCloud<MullsPoint>::Ptr &cloud_unground,
							typename pcl::PointCloud<MullsPoint>::Ptr &cloud_curb,
							int min_grid_pt_num, float grid_resolution, float max_height_difference,
							float neighbor_height_diff, float max_ground_height,
							int ground_random_down_rate, int ground_random_down_down_rate, int nonground_random_down_rate, int reliable_neighbor_grid_num_thre,
							int estimate_ground_normal_method, float normal_estimation_radius, //estimate_ground_normal_method, 0: directly use (0,0,1), 1: estimate normal in fix radius neighborhood , 2: estimate normal in k nearest neighborhood, 3: use ransac to estimate plane coeffs in a grid
							int distance_weight_downsampling_method, float standard_distance,  //standard distance: the distance where the distance_weight is 1
							bool fixed_num_downsampling = false, int down_ground_fixed_num = 1000,
							bool detect_curb_or_not = false, float intensity_thre = FLT_MAX,
							bool apply_grid_wise_outlier_filter = false, float outlier_std_scale = 3.0);

	bool plane_seg_ransac(const typename pcl::PointCloud<MullsPoint>::Ptr &cloud,
						  float threshold, int max_iter, 
						  typename pcl::PointCloud<MullsPoint>::Ptr &planecloud, 
						  pcl::ModelCoefficients::Ptr &coefficients);

	bool estimate_ground_normal_by_ransac(typename pcl::PointCloud<MullsPoint>::Ptr &grid_ground,
										  float dist_thre, int max_iter, float &nx, float &ny, float &nz);

	//Brief: Classfiy the downsampled non-ground points into several types (Pillar, Beam, Facade, Roof, Vertex)
	//according to the pca features (combination of eigen values and eigen vectors)
	bool classify_nground_pts(pcl::PointCloud<MullsPoint>::Ptr &cloud_in,
							  pcl::PointCloud<MullsPoint>::Ptr &cloud_pillar,
							  pcl::PointCloud<MullsPoint>::Ptr &cloud_beam,
							  pcl::PointCloud<MullsPoint>::Ptr &cloud_facade,
							  pcl::PointCloud<MullsPoint>::Ptr &cloud_roof,
							  pcl::PointCloud<MullsPoint>::Ptr &cloud_pillar_down,
							  pcl::PointCloud<MullsPoint>::Ptr &cloud_beam_down,
							  pcl::PointCloud<MullsPoint>::Ptr &cloud_facade_down,
							  pcl::PointCloud<MullsPoint>::Ptr &cloud_roof_down,
							  pcl::PointCloud<MullsPoint>::Ptr &cloud_vertex,
							  float neighbor_searching_radius, int neighbor_k, int neigh_k_min, int pca_down_rate, // one in ${pca_down_rate} unground points would be select as the query points for calculating pca, the else would only be used as neighborhood points
							  float edge_thre, float planar_thre, float edge_thre_down, float planar_thre_down,
							  int extract_vertex_points_method, float curvature_thre, float vertex_curvature_non_max_radius,
							  float linear_vertical_sin_high_thre, float linear_vertical_sin_low_thre,
							  float planar_vertical_sin_high_thre, float planar_vertical_sin_low_thre,
							  bool fixed_num_downsampling = false, int pillar_down_fixed_num = 200, int facade_down_fixed_num = 800, int beam_down_fixed_num = 200,
							  int roof_down_fixed_num = 100, int unground_down_fixed_num = 20000,
							  float beam_height_max = FLT_MAX, float roof_height_min = -FLT_MAX,
							  float feature_pts_ratio_guess = 0.3, bool sharpen_with_nms = true,
							  bool use_distance_adaptive_pca = false);

	//Used in lidar odometry test
	//main entrance to geometric feature points extraction module
	bool extract_semantic_pts(CloudBlockPtr in_block,
							  float vf_downsample_resolution, float gf_grid_resolution,
							  float gf_max_grid_height_diff, float gf_neighbor_height_diff, float gf_max_ground_height,
							  int &gf_down_rate_ground, int &gf_downsample_rate_nonground,
							  float pca_neighbor_radius, int pca_neighbor_k,
							  float edge_thre, float planar_thre, float curvature_thre,
							  float edge_thre_down, float planar_thre_down, bool use_distance_adaptive_pca = false,
							  int distance_inverse_sampling_method = 0, //distance_inverse_downsample, 0: disabled, 1: linear weight, 2: quadratic weight
							  float standard_distance = 15.0,			//the distance where the weight is 1, only useful when distance_inverse_downsample is on
							  int estimate_ground_normal_method = 3,	//estimate_ground_normal_method, 0: directly use (0,0,1), 1: estimate normal in fix radius neighborhood , 2: estimate normal in k nearest neighborhood, 3: use ransac to estimate plane coeffs in a grid
							  float normal_estimation_radius = 2.0,		//only when enabled when estimate_ground_normal_method = 1
							  bool use_adpative_parameters = false, bool apply_scanner_filter = false, bool extract_curb_or_not = false,
							  int extract_vertex_points_method = 2, //use the maximum curvature based keypoints
							  int gf_grid_pt_num_thre = 8, int gf_reliable_neighbor_grid_thre = 0,
							  int gf_down_down_rate_ground = 2, int pca_neighbor_k_min = 8, int pca_down_rate = 1,
							  float intensity_thre = FLT_MAX,														 //default intensity_thre means that highly-reflective objects would not be prefered
							  float linear_vertical_sin_high_thre = 0.94, float linear_vertical_sin_low_thre = 0.17, //70 degree (pillar), 10 degree (beam)
							  float planar_vertical_sin_high_thre = 0.98, float planar_vertical_sin_low_thre = 0.34, //80 degree (roof), 20 degree (facade)
							  bool sharpen_with_nms_on = true, bool fixed_num_downsampling = false, int ground_down_fixed_num = 500,
							  int pillar_down_fixed_num = 200, int facade_down_fixed_num = 800, int beam_down_fixed_num = 200,
							  int roof_down_fixed_num = 200, int unground_down_fixed_num = 20000, float beam_height_max = FLT_MAX, float roof_height_min = 0.0,
							  float approx_scanner_height = 2.0, float underground_thre = -7.0, float feature_pts_ratio_guess = 0.3,
							  bool semantic_assisted = false, bool apply_roi_filtering = false, float roi_min_y = 0.0, float roi_max_y = 0.0);


	//TODO: jingzhao use semantic filter the features
	void filter_with_semantic_mask(CloudBlockPtr in_block, const std::string mask_feature_type = "000000");

	bool get_cloud_pair_intersection(Bounds &intersection_bbx,
									 typename pcl::PointCloud<MullsPoint>::Ptr &pc_ground_tc,
									 typename pcl::PointCloud<MullsPoint>::Ptr &pc_pillar_tc,
									 typename pcl::PointCloud<MullsPoint>::Ptr &pc_beam_tc,
									 typename pcl::PointCloud<MullsPoint>::Ptr &pc_facade_tc,
									 typename pcl::PointCloud<MullsPoint>::Ptr &pc_roof_tc,
									 typename pcl::PointCloud<MullsPoint>::Ptr &pc_vertex_tc,
									 typename pcl::PointCloud<MullsPoint>::Ptr &pc_ground_sc,
									 typename pcl::PointCloud<MullsPoint>::Ptr &pc_pillar_sc,
									 typename pcl::PointCloud<MullsPoint>::Ptr &pc_beam_sc,
									 typename pcl::PointCloud<MullsPoint>::Ptr &pc_facade_sc,
									 typename pcl::PointCloud<MullsPoint>::Ptr &pc_roof_sc,
									 typename pcl::PointCloud<MullsPoint>::Ptr &pc_vertex_sc,
									 bool use_more_points = false);
};

} // namespace mulls

#endif //_INCLUDE_MULLS_FILTER