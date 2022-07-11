#include "pca.h"

#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/pca.h>


namespace mapping_framework
{
namespace common
{

bool PrincipleComponentAnalysis::get_normal_pcar(typename pcl::PointCloud<MullsPoint>::Ptr in_cloud,
												 float radius, pcl::PointCloud<pcl::Normal>::Ptr &normals) {
	// Create the normal estimation class, and pass the input dataset to it;
	pcl::NormalEstimationOMP<MullsPoint, pcl::Normal> ne;
	ne.setNumberOfThreads(omp_get_max_threads()); //More threads sometimes would not speed up the procedure
	ne.setInputCloud(in_cloud);
	// Create an empty kd-tree representation, and pass it to the normal estimation object;
	typename pcl::search::KdTree<MullsPoint>::Ptr tree(new pcl::search::KdTree<MullsPoint>());
	ne.setSearchMethod(tree);
	// Use all neighbors in a sphere of radius;
	ne.setRadiusSearch(radius);
	// Compute the normal
	ne.compute(*normals);
	check_normal(normals);
	return true;
}

bool PrincipleComponentAnalysis::get_normal_pcak(typename pcl::PointCloud<MullsPoint>::Ptr in_cloud,
												 int K, pcl::PointCloud<pcl::Normal>::Ptr &normals) {
	// Create the normal estimation class, and pass the input dataset to it;
	pcl::NormalEstimationOMP<MullsPoint, pcl::Normal> ne;
	ne.setNumberOfThreads(omp_get_max_threads()); //More threads sometimes would not speed up the procedure
	ne.setInputCloud(in_cloud);
	// Create an empty kd-tree representation, and pass it to the normal estimation object;
	typename pcl::search::KdTree<MullsPoint>::Ptr tree(new pcl::search::KdTree<MullsPoint>());
	ne.setSearchMethod(tree);
	// Use all neighbors in a sphere of radius;
	ne.setKSearch(K);
	// Compute the normal
	ne.compute(*normals);
	check_normal(normals);
	return true;
}

bool PrincipleComponentAnalysis::get_pc_pca_feature(typename pcl::PointCloud<MullsPoint>::Ptr in_cloud, std::vector<pca_feature_t> &features, 
                                                    typename pcl::KdTreeFLANN<MullsPoint>::Ptr &tree, float radius, int nearest_k, 
													int min_k, int pca_down_rate, bool distance_adaptive_on, float unit_dist) {
	features.resize(in_cloud->points.size());
	omp_set_num_threads(std::min(6, omp_get_max_threads()));
#pragma omp parallel for												 //Multi-thread
	for (size_t i = 0u; i < in_cloud->points.size(); i += pca_down_rate) //faster way
	{
		std::vector<int> search_indices_used; //points would be stored in sequence (from the closest point to the farthest point within the neighborhood)
		std::vector<int> search_indices;	  //point index vector
		std::vector<float> squared_distances; //distance vector

		float neighborhood_r = radius;
		int neighborhood_k = nearest_k;

		if (distance_adaptive_on) {
			double dist = std::sqrt(in_cloud->points[i].x * in_cloud->points[i].x +
									in_cloud->points[i].y * in_cloud->points[i].y +
									in_cloud->points[i].z * in_cloud->points[i].z);
			if (dist > unit_dist) {
				neighborhood_r = std::sqrt(dist / unit_dist) * radius;
			}
		}
		//nearest_k=0 --> the knn is disabled, only the rnn is used
		tree->radiusSearch(i, neighborhood_r, search_indices, squared_distances, neighborhood_k);

		features[i].pt.x = in_cloud->points[i].x;
		features[i].pt.y = in_cloud->points[i].y;
		features[i].pt.z = in_cloud->points[i].z;
		features[i].ptId = i;
		features[i].pt_num = search_indices.size();

		//deprecated
		features[i].close_to_query_point.resize(search_indices.size());
		for (size_t j = 0u; j < search_indices.size(); j++) {
			if (squared_distances[j] < 0.64 * radius * radius) // 0.5^(2/3)
				features[i].close_to_query_point[j] = true;
			else
				features[i].close_to_query_point[j] = false;
		}

		get_pca_feature(in_cloud, search_indices, features[i]);

		if (features[i].pt_num > min_k)
			assign_normal(in_cloud->points[i], features[i]);
		std::vector<int>().swap(search_indices);
		std::vector<int>().swap(search_indices_used);
		std::vector<float>().swap(squared_distances);
	}
	return true;
}

bool PrincipleComponentAnalysis::get_pca_feature(typename pcl::PointCloud<MullsPoint>::Ptr in_cloud,
												 std::vector<int> &search_indices, pca_feature_t &feature) {
	size_t pt_num = search_indices.size();
	if (pt_num <= 3u)
		return false;

	typename pcl::PointCloud<MullsPoint>::Ptr selected_cloud(new pcl::PointCloud<MullsPoint>());
	for (size_t i = 0u; i < pt_num; ++i)
		selected_cloud->points.push_back(in_cloud->points[search_indices[i]]);

	pcl::PCA<MullsPoint> pca_operator;
	pca_operator.setInputCloud(selected_cloud);

	// Compute eigen values and eigen vectors
	Eigen::Matrix3f eigen_vectors = pca_operator.getEigenVectors();
	Eigen::Vector3f eigen_values = pca_operator.getEigenValues();

	feature.vectors.principalDirection = eigen_vectors.col(0);
	feature.vectors.normalDirection = eigen_vectors.col(2);

	feature.vectors.principalDirection.normalize();
	feature.vectors.normalDirection.normalize();

	feature.values.lamada1 = eigen_values(0);
	feature.values.lamada2 = eigen_values(1);
	feature.values.lamada3 = eigen_values(2);

	if ((feature.values.lamada1 + feature.values.lamada2 + feature.values.lamada3) == 0)
		feature.curvature = 0;
	else
		feature.curvature = feature.values.lamada3 / (feature.values.lamada1 + feature.values.lamada2 + feature.values.lamada3);

	feature.linear_2 = ((feature.values.lamada1) - (feature.values.lamada2)) / (feature.values.lamada1);
	feature.planar_2 = ((feature.values.lamada2) - (feature.values.lamada3)) / (feature.values.lamada1);
	feature.spherical_2 = (feature.values.lamada3) / (feature.values.lamada1);

	search_indices.swap(feature.neighbor_indices);
	return true;
}

bool PrincipleComponentAnalysis::assign_normal(MullsPoint &pt, pca_feature_t &pca_feature, bool is_plane_feature) {
	if (is_plane_feature) {
		pt.normal_x = pca_feature.vectors.normalDirection.x();
		pt.normal_y = pca_feature.vectors.normalDirection.y();
		pt.normal_z = pca_feature.vectors.normalDirection.z();
		pt.data_c[3] = pca_feature.planar_2; //planrity
	} else {
		pt.normal_x = pca_feature.vectors.principalDirection.x();
		pt.normal_y = pca_feature.vectors.principalDirection.y();
		pt.normal_z = pca_feature.vectors.principalDirection.z();
		pt.data_c[3] = pca_feature.linear_2; //linarity
	}
	return true;
}

void PrincipleComponentAnalysis::check_normal(pcl::PointCloud<pcl::Normal>::Ptr &normals) {
	//It is advisable to check the normals before the call to compute()
	for (size_t i = 0u; i < normals->points.size(); i++) {
		if (!pcl::isFinite<pcl::Normal>(normals->points[i])) {
			normals->points[i].normal_x = 0.577; // 1/ sqrt(3)
			normals->points[i].normal_y = 0.577;
			normals->points[i].normal_z = 0.577;
		}
	}
}

} // namespace common
} // namespace mapping_framework
