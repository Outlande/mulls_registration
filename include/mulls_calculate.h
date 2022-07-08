#ifndef _INCLUDE_MULLS_REG_HPP
#define _INCLUDE_MULLS_REG_HPP

#include <math.h>

//pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>
#include <pcl/registration/correspondence_rejection_features.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/registration/correspondence_rejection_var_trimmed.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls_weighted.h>
#include <pcl/registration/transformation_estimation_point_to_plane_weighted.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/ia_ransac.h>

#if TEASER_ON
//teaser++ (global registration)
#include <teaser/ply_io.h>
#include <teaser/registration.h>
#include <teaser/certification.h>
#endif

#include <chrono>
#include <limits>
#include <time.h>

#include "mulls_filter.h"
#include "mulls_util.h"
#include "pca.h"
#include <glog/logging.h>

namespace mulls
{

class MullsCalculate
{
  public:
	//brief: Compute fpfh_feature
	void compute_fpfh_feature(const pcl::PointCloud<MullsPoint>::Ptr &input_cloud,
							  fpfhPtr &cloud_fpfh, float search_radius);

	//brief: Accomplish Coarse registration using FPFH SAC
	double coarse_reg_fpfhsac(const pcl::PointCloud<MullsPoint>::Ptr &source_cloud,
							  const pcl::PointCloud<MullsPoint>::Ptr &target_cloud,
							  pcl::PointCloud<MullsPoint>::Ptr &traned_source,
							  Eigen::Matrix4d &transformationS2T, float search_radius);
  
	//NCC: neighborhood category context descriptor
	bool find_feature_correspondence_ncc(const pcl::PointCloud<MullsPoint>::Ptr &target_kpts, const pcl::PointCloud<MullsPoint>::Ptr &source_kpts,
										 pcl::PointCloud<MullsPoint>::Ptr &target_corrs, pcl::PointCloud<MullsPoint>::Ptr &source_corrs,
										 bool fixed_num_corr = false, int corr_num = 2000, bool reciprocal_on = true);

	//coarse global registration using RANSAC 
	int coarse_reg_ransac(const pcl::PointCloud<MullsPoint>::Ptr &target_pts,
						  const pcl::PointCloud<MullsPoint>::Ptr &source_pts,
						  Eigen::Matrix4d &tran_mat, float noise_bound = 0.2, 
						  int min_inlier_num = 8, int max_iter_num = 20000);

	//coarse global registration using TEASER ++  (faster and more robust to outlier than RANSAC)
	int coarse_reg_teaser(const pcl::PointCloud<MullsPoint>::Ptr &target_pts,
						  const pcl::PointCloud<MullsPoint>::Ptr &source_pts,
						  Eigen::Matrix4d &tran_mat, float noise_bound = 0.2, int min_inlier_num = 8);

	bool determine_source_target_cloud(const CloudBlockPtr &block_1, const CloudBlockPtr &block_2, Constraint &registration_cons);

	//--------------------------------------------------------------------------------------------------------------------------//
	//Multi-metrics Linear Least Square ICP (MULLS-ICP)
	//An efficient cross-template point cloud fine registration method
	//Author: Yue Pan
	//Outlines:
	//1.Preprocessing: Downsample the raw point cloud and classify it into several categories
	//  (Planar: Ground, Roof, Facade, Linear: Pillar, Beam, Sphere: Vertex), refer to the extract_semantic_pts function in 'filter.hpp'
	//2.Find correspondences within the same category with a trimmed strategy
	//3.Estimate the transformation that minimizes the weighted (x,y,z balanced) distance after applying the transformation,
	//  We use the point-to-point, point-to-line and point-to-plane distance metrics for Sphere, Linear and Planar points respectivly.
	//  This is sloved very efficiently by Linear Least Square (take tx,ty,tz,roll ,pitch,yaw as unknowns)
	//4.Update the Source Point Cloud and keep iterating
	//5.Till converge, Output the 4*4 final transformation matrix and the 6*6 information matrix
	//TODO polish the codes --> delete some unnecssary codes and also encapsulate some codes in private functions
	int mm_lls_icp(Constraint &registration_cons, // cblock_1 (target point cloud), cblock_2 (source point cloud)
				   int max_iter_num = 20, float dis_thre_unit = 1.5,
				   float converge_translation = 0.002, float converge_rotation_d = 0.01,
				   float dis_thre_min = 0.4, float dis_thre_update_rate = 1.1, std::string used_feature_type = "111110",
				   std::string weight_strategy = "1101", float z_xy_balanced_ratio = 1.0,
				   float pt2pt_residual_window = 0.1, float pt2pl_residual_window = 0.1, float pt2li_residual_window = 0.1,
				   Eigen::Matrix4d initial_guess = Eigen::Matrix4d::Identity(), //used_feature_type (1: on, 0: off, order: ground, pillar, facade, beam, roof, vetrex)
				   bool apply_intersection_filter = true, bool apply_motion_undistortion_while_registration = false,
				   bool normal_shooting_on = false, float normal_bearing = 45.0, bool use_more_points = false,
				   bool keep_less_source_points = false, float sigma_thre = 0.5, float min_neccessary_corr_ratio = 0.03, float max_bearable_rotation_d = 45.0);
	//sigma_thre means the maximum threshold of the posterior standar deviation of the registration LLS (unit:m) 
  protected:
  private:
	void batch_transform_feature_points(pcl::PointCloud<MullsPoint>::Ptr pc_ground, pcl::PointCloud<MullsPoint>::Ptr pc_pillar,
										pcl::PointCloud<MullsPoint>::Ptr pc_beam, pcl::PointCloud<MullsPoint>::Ptr pc_facade,
										pcl::PointCloud<MullsPoint>::Ptr pc_roof, pcl::PointCloud<MullsPoint>::Ptr pc_vertex,
										Eigen::Matrix4d &Tran);

	//Time complexity of kdtree (in this case, the target point cloud [n points] is used for construct the tree while each point in source point cloud acts as a query point)
	//build tree: O(nlogn) ---> so it's better to build the tree only once
	//searching 1-nearest neighbor: O(logn) in average ---> so we can bear a larger number of target points
	bool determine_corres(pcl::PointCloud<MullsPoint>::Ptr &Source_Cloud, pcl::PointCloud<MullsPoint>::Ptr &Target_Cloud,
						  const pcl::search::KdTree<MullsPoint>::Ptr &target_kdtree, float dis_thre,
						  boost::shared_ptr<pcl::Correspondences> &Corr_f, bool normal_shooting_on, bool normal_check = true,
						  float angle_thre_degree = 40, bool duplicate_check = true, int K_filter_distant_point = 500);
	
	void update_corr_dist_thre(float &dis_thre_ground, float &dis_thre_pillar, float &dis_thre_beam,
							   float &dis_thre_facade, float &dis_thre_roof, float &dis_thre_vertex,
							   float dis_thre_update_rate, float dis_thre_min);

	//brief: entrance to mulls transformation estimation
	bool multi_metrics_lls_tran_estimation(const pcl::PointCloud<MullsPoint>::Ptr &Source_Ground, const pcl::PointCloud<MullsPoint>::Ptr &Target_Ground, boost::shared_ptr<pcl::Correspondences> &Corr_Ground,
										   const pcl::PointCloud<MullsPoint>::Ptr &Source_Pillar, const pcl::PointCloud<MullsPoint>::Ptr &Target_Pillar, boost::shared_ptr<pcl::Correspondences> &Corr_Pillar,
										   const pcl::PointCloud<MullsPoint>::Ptr &Source_Beam, const pcl::PointCloud<MullsPoint>::Ptr &Target_Beam, boost::shared_ptr<pcl::Correspondences> &Corr_Beam,
										   const pcl::PointCloud<MullsPoint>::Ptr &Source_Facade, const pcl::PointCloud<MullsPoint>::Ptr &Target_Facade, boost::shared_ptr<pcl::Correspondences> &Corr_Facade,
										   const pcl::PointCloud<MullsPoint>::Ptr &Source_Roof, const pcl::PointCloud<MullsPoint>::Ptr &Target_Roof, boost::shared_ptr<pcl::Correspondences> &Corr_Roof,
										   const pcl::PointCloud<MullsPoint>::Ptr &Source_Vertex, const pcl::PointCloud<MullsPoint>::Ptr &Target_Vertex, boost::shared_ptr<pcl::Correspondences> &Corr_Vertex,
										   Vector6d &unknown_x, Matrix6d &cofactor_matrix, int iter_num, std::string weight_strategy, float z_xy_balance_ratio = 1.0,
										   float pt2pt_residual_window = 0.1, float pt2pl_residual_window = 0.1, float pt2li_residual_window = 0.1);

	//Linearization of Rotation Matrix
	//R = I + (alpha, beta, gamma) ^
	//  = | 1      -gamma    beta |
	//    | gamma   1       -alpha|
	//    |-beta    alpha     1   |

	//point-to-point LLS
	bool pt2pt_lls_summation(const pcl::PointCloud<MullsPoint>::Ptr &Source_Cloud, const pcl::PointCloud<MullsPoint>::Ptr &Target_Cloud,
							 boost::shared_ptr<pcl::Correspondences> &Corr, Matrix6d &ATPA, Vector6d &ATPb, int iter_num,
							 float weight, bool dist_weight_or_not = false, bool residual_weight_or_not = false,
							 bool intensity_weight_or_not = false,
							 float residual_window_size = 0.1);

	//point-to-plane LLS
	bool pt2pl_lls_summation(const pcl::PointCloud<MullsPoint>::Ptr &Source_Cloud, const pcl::PointCloud<MullsPoint>::Ptr &Target_Cloud,
							 boost::shared_ptr<pcl::Correspondences> &Corr, Matrix6d &ATPA, Vector6d &ATPb, int iter_num,
							 float weight, bool dist_weight_or_not = false, bool residual_weight_or_not = false,
							 bool intensity_weight_or_not = false,
							 float residual_window_size = 0.1);

	//point-to-line LLS (calculated using primary direction vector), used now
	//the normal vector here actually stores the primary direcyion vector (for easier calculation of the residual)
	bool pt2li_lls_pri_direction_summation(const pcl::PointCloud<MullsPoint>::Ptr &Source_Cloud, const pcl::PointCloud<MullsPoint>::Ptr &Target_Cloud,
										   boost::shared_ptr<pcl::Correspondences> &Corr, Matrix6d &ATPA, Vector6d &ATPb, int iter_num,
										   float weight, bool dist_weight_or_not = false, bool residual_weight_or_not = false,
										   bool intensity_weight_or_not = false,
										   float residual_window_size = 0.1);

	bool ground_3dof_lls_tran_estimation(const pcl::PointCloud<MullsPoint>::Ptr &Source_Ground,
										 const pcl::PointCloud<MullsPoint>::Ptr &Target_Ground,
										 boost::shared_ptr<pcl::Correspondences> &Corr_Ground,
										 Eigen::Vector3d &unknown_x, Eigen::Matrix3d &cofactor_matrix,
										 int iter_num, std::string weight_strategy);

	//ground 3dof : roll, pitch, z
	bool pt2pl_ground_3dof_lls_summation(const pcl::PointCloud<MullsPoint>::Ptr &Source_Cloud, const pcl::PointCloud<MullsPoint>::Ptr &Target_Cloud,
										 boost::shared_ptr<pcl::Correspondences> &Corr, Eigen::Matrix3d &ATPA, Eigen::Vector3d &ATPb, int iter_num,
										 float weight, bool dist_weight_or_not = false, bool residual_weight_or_not = false, bool intensity_weight_or_not = false,
										 float residual_window_size = 0.1);

	//calculate residual v
	bool get_multi_metrics_lls_residual(const pcl::PointCloud<MullsPoint>::Ptr &Source_Ground, const pcl::PointCloud<MullsPoint>::Ptr &Target_Ground, boost::shared_ptr<pcl::Correspondences> &Corr_Ground,
										const pcl::PointCloud<MullsPoint>::Ptr &Source_Pillar, const pcl::PointCloud<MullsPoint>::Ptr &Target_Pillar, boost::shared_ptr<pcl::Correspondences> &Corr_Pillar,
										const pcl::PointCloud<MullsPoint>::Ptr &Source_Beam, const pcl::PointCloud<MullsPoint>::Ptr &Target_Beam, boost::shared_ptr<pcl::Correspondences> &Corr_Beam,
										const pcl::PointCloud<MullsPoint>::Ptr &Source_Facade, const pcl::PointCloud<MullsPoint>::Ptr &Target_Facade, boost::shared_ptr<pcl::Correspondences> &Corr_Facade,
										const pcl::PointCloud<MullsPoint>::Ptr &Source_Roof, const pcl::PointCloud<MullsPoint>::Ptr &Target_Roof, boost::shared_ptr<pcl::Correspondences> &Corr_Roof,
										const pcl::PointCloud<MullsPoint>::Ptr &Source_Vertex, const pcl::PointCloud<MullsPoint>::Ptr &Target_Vertex, boost::shared_ptr<pcl::Correspondences> &Corr_Vertex,
										const Vector6d &transform_x, double &sigma_square_post, double sigma_thre = 0.2);

	bool pt2pt_lls_residual(const pcl::PointCloud<MullsPoint>::Ptr &Source_Cloud, const pcl::PointCloud<MullsPoint>::Ptr &Target_Cloud,
							boost::shared_ptr<pcl::Correspondences> &Corr, const Vector6d &transform_x, double &VTPV, int &observation_count);

	bool pt2pl_lls_residual(const pcl::PointCloud<MullsPoint>::Ptr &Source_Cloud, const pcl::PointCloud<MullsPoint>::Ptr &Target_Cloud,
							boost::shared_ptr<pcl::Correspondences> &Corr, const Vector6d &transform_x, double &VTPV, int &observation_count);

	//primary vector stored as point normal
	bool pt2li_lls_residual(const pcl::PointCloud<MullsPoint>::Ptr &Source_Cloud, const pcl::PointCloud<MullsPoint>::Ptr &Target_Cloud,
							boost::shared_ptr<pcl::Correspondences> &Corr, const Vector6d &transform_x, double &VTPV, int &observation_count);

	// the following lines contain the various weighting functions: distance weight, intensity-compatible weight and residual weight
	// Intuition: three part, near , medium , far
	// near used to control translation
	// far used to control rotation
	// medium used to control both
	//TODO Mulls: change the distance weight according to the iteration number (mathematic deducing)
	float get_weight_by_dist_adaptive(float dist, int iter_num, float unit_dist = 30.0,
	                                 float b_min = 0.7, float b_max = 1.3, float b_step = 0.05);

	//standard
	//unit_dist = 60.0 (is just a multiplier constant) 
	inline float get_weight_by_dist(float dist, float unit_dist = 60.0, float base_value = 0.7);

	inline float get_weight_by_intensity(float intensity_1, float intensity_2, float base_value = 0.6, 
	                                     float intensity_scale = 255.0);

	//By huber loss function
	//test different kind of robust kernel function here
	inline float get_weight_by_residual(float res, float huber_thre = 0.05, int delta = 1);

	//general function for m-estimation
	//test different kind of robust kernel function here
	float get_weight_by_residual_general(float res, float thre = 0.05, float alpha = 2.0);

	//roll - pitch - yaw rotation (x - y' - z'') ----> this is for our tiny angle approximation
	bool construct_trans_a(const double &tx, const double &ty, const double &tz,
						   const double &alpha, const double &beta, const double &gamma,
						   Eigen::Matrix4d &transformation_matrix);

	//Brief: calculate the Jacobi Matrix of the imaginary part of a quaternion (q1,q2,q3) with regard to its corresponding euler angle (raw,pitch,yaw)
	//for converting the euler angle variance-covariance matrix to quaternion variance-covariance matrix using variance-covariance propagation law
	//Log: Pay attetion to the euler angle order here. I originally use yaw, pitch, roll (z, y', x'') here, the correct one should be roll, pitch, yaw (x, y', z'')
	//reference:
	//1 . http://easyspin.org/easyspin/documentation/eulerangles.html
	//2 . https://en.wikipedia.org/wiki/Euler_angles
	bool get_quat_euler_jacobi(const Eigen::Vector3d &euler_angle, Eigen::Matrix3d &Jacobi);

	//this function is for speed up the registration process when the point number is a bit too big
	bool keep_less_source_pts(pcl::PointCloud<MullsPoint>::Ptr &pc_ground_tc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_pillar_tc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_beam_tc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_facade_tc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_roof_tc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_vertex_tc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_ground_sc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_pillar_sc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_beam_sc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_facade_sc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_roof_sc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_vertex_sc,
							  int ground_down_rate = 4, int facade_down_rate = 2, 
							  int target_down_rate = 2);

	bool intersection_filter( Constraint &registration_cons,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_ground_tc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_pillar_tc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_beam_tc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_facade_tc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_roof_tc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_vertex_tc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_ground_sc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_pillar_sc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_beam_sc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_facade_sc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_roof_sc,
							  pcl::PointCloud<MullsPoint>::Ptr &pc_vertex_sc,
							  float bbx_pad = 1.0);
};

} // namespace mulls

#endif //_INCLUDE_MULLS_REG_HPP
