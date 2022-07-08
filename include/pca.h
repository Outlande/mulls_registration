#ifndef _INCLUDE_MULLS_PCA_
#define _INCLUDE_MULLS_PCA_

//pcl
#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/pca.h>
#include <glog/logging.h>
#include <chrono>
#include <limits>
#include <time.h>

#include <vector>

#include "mulls_util.h"

namespace mulls
{

template <typename PointT>
class PrincipleComponentAnalysis
{
  public:
	/**
	* \brief Estimate the normals of the input Point Cloud by PCL speeding up with OpenMP
	* \param[in] in_cloud is the input Point Cloud Pointer
	* \param[in] radius is the neighborhood search radius (m) for KD Tree
	* \param[out] normals is the normal of all the points from the Point Cloud
	*/
	bool get_normal_pcar(typename pcl::PointCloud<PointT>::Ptr in_cloud, float radius,
						 pcl::PointCloud<pcl::Normal>::Ptr &normals);

	bool get_pc_normal_pcar(typename pcl::PointCloud<PointT>::Ptr in_cloud, float radius,
							pcl::PointCloud<pcl::PointNormal>::Ptr &pointnormals);

	bool get_normal_pcak(typename pcl::PointCloud<PointT>::Ptr in_cloud, int K,
						 pcl::PointCloud<pcl::Normal>::Ptr &normals);

	bool get_pc_normal_pcak(typename pcl::PointCloud<PointT>::Ptr in_cloud, int K,
							pcl::PointCloud<pcl::PointNormal>::Ptr &pointnormals);
	
	/**
	* \brief Principle Component Analysis (PCA) of the Point Cloud with fixed search radius
	* \param[in] in_cloud is the input Point Cloud (XYZI) Pointer
	* \param[in]     radius is the neighborhood search radius (m) for KD Tree
	* \param[out]features is the pca_feature_t vector of all the points from the Point Cloud
	*/

	// radius neighborhood
	bool get_pc_pca_feature(typename pcl::PointCloud<PointT>::Ptr in_cloud,
							std::vector<pca_feature_t> &features, float radius);
	
	// K neighborhood
	bool get_pc_pca_feature(typename pcl::PointCloud<PointT>::Ptr in_cloud,
							std::vector<pca_feature_t> &features, int K);

	// R - K neighborhood (without already built-kd tree)
	//within the radius, we would select the nearest K points for calculating PCA
	bool get_pc_pca_feature(typename pcl::PointCloud<PointT>::Ptr in_cloud,
							std::vector<pca_feature_t> &features,
							float radius, int nearest_k, int min_k = 1, int pca_down_rate = 1,
							bool distance_adaptive_on = false, float unit_dist = 35.0);

	// R - K neighborhood (with already built-kd tree)
	//within the radius, we would select the nearest K points for calculating PCA
	bool get_pc_pca_feature(typename pcl::PointCloud<PointT>::Ptr in_cloud,
							std::vector<pca_feature_t> &features, typename pcl::KdTreeFLANN<PointT>::Ptr &tree,
							float radius, int nearest_k, int min_k = 1, int pca_down_rate = 1,
							bool distance_adaptive_on = false, float unit_dist = 35.0);

	void calculate_normal_inconsistency(typename pcl::PointCloud<PointT>::Ptr in_cloud,
										std::vector<pca_feature_t> &features);

	/**
	* \brief Use PCL to accomplish the Principle Component Analysis (PCA)
	* of one point and its neighborhood
	* \param[in] in_cloud is the input Point Cloud Pointer
	* \param[in] search_indices is the neighborhood points' indices of the search point.
	* \param[out]feature is the pca_feature_t of the search point.
	*/
	bool get_pca_feature(typename pcl::PointCloud<PointT>::Ptr in_cloud,
						 std::vector<int> &search_indices, pca_feature_t &feature);

	//is_palne_feature (true: assign point normal as pca normal vector, false: assign point normal as pca primary direction vector)
	bool assign_normal(PointT &pt, pca_feature_t &pca_feature, bool is_plane_feature = true);

  protected:
  private:
	/**
	* \brief Check the Normals (if they are finite)
	* \param normals is the input Point Cloud (XYZI)'s Normal Pointer
	*/
	void check_normal(pcl::PointCloud<pcl::Normal>::Ptr &normals);
};

} // namespace mulls
#endif //_INCLUDE_MULLS_PCA_
