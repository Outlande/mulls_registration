#include "mulls_filter.h"

#include <cfloat>
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
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>

namespace mapping_framework
{
namespace common
{
bool MullsFilter::xy_normal_balanced_downsample(pcl::PointCloud<MullsPoint>::Ptr &cloud_in_out,
									            int keep_number_per_sector, int sector_num) {
	if (cloud_in_out->points.size() <= size_t(keep_number_per_sector))
		return false;

	pcl::PointCloud<MullsPoint>::Ptr cloud_temp(new pcl::PointCloud<MullsPoint>);

	//divide into ${sector_num} sectors according to normal_x and normal_y
	std::vector<pcl::PointCloud<MullsPoint>::Ptr> sectors(sector_num);

	//initialization
	for (int j = 0; j < sector_num; j++)
		sectors[j].reset(new pcl::PointCloud<MullsPoint>());

	double angle_per_sector = 360.0 / sector_num;

	for (size_t i = 0u; i < cloud_in_out->points.size(); i++)
	{
		double ang = std::atan2(cloud_in_out->points[i].normal_y, cloud_in_out->points[i].normal_x);
		//atan2 (x,y)  --> [-pi , pi]
		//I:   x+ y+ --> [0,   pi/2]
		//II:  x- y+ --> [pi/2,  pi]
		//III: x- y- --> [-pi,-pi/2]
		//IV:  x+ y- --> [-pi/2,  0]

		if (ang < 0)
			ang += 2 * M_PI; // --> transform to the anti-clockwise angle from +x axis

		ang *= (180.0 / M_PI);

		int sector_id = (int)(ang / angle_per_sector);
		sectors[sector_id]->points.push_back(cloud_in_out->points[i]); //push_back may cause collision for multi-thread processing
	}

	for (int j = 0; j < sector_num; j++)
	{
		random_downsample_pcl(sectors[j], keep_number_per_sector);
		cloud_temp->points.insert(cloud_temp->points.end(), sectors[j]->points.begin(), sectors[j]->points.end());
	}

	cloud_temp->points.swap(cloud_in_out->points);
	std::vector<pcl::PointCloud<MullsPoint>::Ptr>().swap(sectors);
	return true;
}

bool MullsFilter::random_downsample_pcl(pcl::PointCloud<MullsPoint>::Ptr &cloud_in_out, int keep_number) {
	if (cloud_in_out->points.size() <= size_t(keep_number))
		return false;
	else {
		if (keep_number == 0) {
			cloud_in_out.reset(new pcl::PointCloud<MullsPoint>());
			return false;
		} else {
			pcl::PointCloud<MullsPoint>::Ptr cloud_temp(new pcl::PointCloud<MullsPoint>);
			pcl::RandomSample<MullsPoint> ran_sample(true); // Extract removed indices
			ran_sample.setInputCloud(cloud_in_out);
			ran_sample.setSample(keep_number);
			ran_sample.filter(*cloud_temp);
			cloud_temp->points.swap(cloud_in_out->points);
			return true;
		}
	}
}


bool MullsFilter::random_downsample_pcl(pcl::PointCloud<MullsPoint>::Ptr &cloud_in,
							            pcl::PointCloud<MullsPoint>::Ptr &cloud_out, 
										int keep_number) {
	if (cloud_in->points.size() <= size_t(keep_number)) {
		cloud_out = cloud_in;
		return false;
	} else {
		if (keep_number == 0)
			return false;
		else {
			pcl::RandomSample<MullsPoint> ran_sample(true); // Extract removed indices
			ran_sample.setInputCloud(cloud_in);
			ran_sample.setSample(keep_number);
			ran_sample.filter(*cloud_out);
			return true;
		}
	}
}

bool MullsFilter::random_downsample(const pcl::PointCloud<MullsPoint>::Ptr &cloud_in,
						            pcl::PointCloud<MullsPoint>::Ptr &cloud_out, 
									int downsample_ratio) {
	if (downsample_ratio > 1) {
		cloud_out->points.clear();
		for (size_t i = 0u; i < cloud_in->points.size(); i++) {
			if (i % downsample_ratio == 0)
				cloud_out->points.push_back(cloud_in->points[i]);
		}
		return true;
	}
	else
		return false;
}

bool MullsFilter::random_downsample(pcl::PointCloud<MullsPoint>::Ptr &cloud_in_out, int downsample_ratio) {
	pcl::PointCloud<MullsPoint>::Ptr cloud_temp(new pcl::PointCloud<MullsPoint>);
	if (downsample_ratio > 1) {
		for (size_t i = 0u; i < cloud_in_out->points.size(); i++) {
			if (i % downsample_ratio == 0)
				cloud_temp->points.push_back(cloud_in_out->points[i]);
		}
		cloud_temp->points.swap(cloud_in_out->points);
		return true;
	}
	else
		return false;
}


bool MullsFilter::scanner_filter(const pcl::PointCloud<MullsPoint>::Ptr &cloud_in_out, float self_radius,
                                 float ghost_radius, float z_min_thre_ghost, float z_min_thre_global) {
	pcl::PointCloud<MullsPoint>::Ptr cloud_temp(new pcl::PointCloud<MullsPoint>);
	for (size_t i = 0u; i < cloud_in_out->points.size(); i++) {
		float dis_square = cloud_in_out->points[i].x * cloud_in_out->points[i].x + cloud_in_out->points[i].y * cloud_in_out->points[i].y;
		if (dis_square > self_radius * self_radius && cloud_in_out->points[i].z > z_min_thre_global) {
			if (dis_square > ghost_radius * ghost_radius || cloud_in_out->points[i].z > z_min_thre_ghost)
				cloud_temp->points.push_back(cloud_in_out->points[i]);
		}
	}
	cloud_temp->points.swap(cloud_in_out->points);
	return true;
}

bool MullsFilter::bbx_filter(const pcl::PointCloud<MullsPoint>::Ptr &cloud_in,
				             pcl::PointCloud<MullsPoint>::Ptr &cloud_out, Bounds &bbx) {
	for (size_t i = 0u; i < cloud_in->points.size(); i++) {
		//In the bounding box
		if (cloud_in->points[i].x > bbx.min_x && cloud_in->points[i].x < bbx.max_x &&
			cloud_in->points[i].y > bbx.min_y && cloud_in->points[i].y < bbx.max_y &&
			cloud_in->points[i].z > bbx.min_z && cloud_in->points[i].z < bbx.max_z) {
			cloud_out->points.push_back(cloud_in->points[i]);
		}
	}
	return true;
}

bool MullsFilter::bbx_filter(const pcl::PointCloud<MullsPoint>::Ptr &cloud_in_out, Bounds &bbx, bool delete_box) {
	pcl::PointCloud<MullsPoint>::Ptr cloud_temp(new pcl::PointCloud<MullsPoint>);
	int original_pts_num = cloud_in_out->points.size();
	for (size_t i = 0u; i < cloud_in_out->points.size(); i++) {
		//In the bounding box
		if (cloud_in_out->points[i].x > bbx.min_x && cloud_in_out->points[i].x < bbx.max_x &&
			cloud_in_out->points[i].y > bbx.min_y && cloud_in_out->points[i].y < bbx.max_y &&
			cloud_in_out->points[i].z > bbx.min_z && cloud_in_out->points[i].z < bbx.max_z) {
			if (!delete_box)
				cloud_temp->points.push_back(cloud_in_out->points[i]);
		} else {
			if (delete_box)
				cloud_temp->points.push_back(cloud_in_out->points[i]);
		}
	}
	cloud_temp->points.swap(cloud_in_out->points);
	return true;
}

bool MullsFilter::encode_stable_points(const pcl::PointCloud<MullsPoint>::Ptr &cloud_in,
									   pcl::PointCloud<MullsPoint>::Ptr &cloud_out,
									   const std::vector<pca_feature_t> &features,
									   const std::vector<int> &index_with_feature,
									   float min_curvature,
									   int min_feature_point_num_neighborhood,
									   int min_point_num_neighborhood) {
	for (size_t i = 0; i < features.size(); ++i)
	{
		if(features[i].pt_num > min_point_num_neighborhood && features[i].curvature > min_curvature) {
			float accu_intensity = 0.0;
			MullsPoint pt;
			pt = cloud_in->points[i];
			pt.data_c[3] = features[i].curvature; //save in data_c[3]

			int neighbor_total_count = 0, pillar_count = 0, beam_count = 0, facade_count = 0, roof_count = 0;
			int pillar_close_count = 0, pillar_far_count = 0, beam_close_count = 0, beam_far_count = 0, facade_close_count = 0, facade_far_count = 0, roof_close_count = 0, roof_far_count = 0;

			neighbor_total_count = features[i].neighbor_indices.size();

			for (int j = 0; j < neighbor_total_count; j++) {
				int temp_neighbor_index = features[i].neighbor_indices[j];
				switch (index_with_feature[temp_neighbor_index]) {
				case 1: {
					pillar_count++;
					if (features[i].close_to_query_point[j])
						pillar_close_count++;
					else
						pillar_far_count++;
					break;
				}
				case 2: {
					beam_count++;
					if (features[i].close_to_query_point[j])
						beam_close_count++;
					else
						beam_far_count++;
					break;
				}
				case 3: {
					facade_count++;
					if (features[i].close_to_query_point[j])
						facade_close_count++;
					else
						facade_far_count++;
					break;
				}
				case 4: {
					roof_count++;
					if (features[i].close_to_query_point[j])
						roof_close_count++;
					else
						roof_far_count++;
					break;
				}
				default:
					break;
				}
				accu_intensity += cloud_in->points[temp_neighbor_index].intensity;
			}
			if (pillar_count + beam_count + facade_count + roof_count < min_feature_point_num_neighborhood)
				continue;

			//TODO MULLS: it's a very stupid way to doing so, change the feature encoding in code refactoring
			pillar_count = 100 * pillar_count / neighbor_total_count;
			beam_count = 100 * beam_count / neighbor_total_count;
			facade_count = 100 * facade_count / neighbor_total_count;
			roof_count = 100 * roof_count / neighbor_total_count;
			pillar_close_count = 100 * pillar_close_count / neighbor_total_count;
			beam_close_count = 100 * beam_close_count / neighbor_total_count;
			facade_close_count = 100 * facade_close_count / neighbor_total_count;
			roof_close_count = 100 * roof_close_count / neighbor_total_count;
			pillar_far_count = 100 * pillar_far_count / neighbor_total_count;
			beam_far_count = 100 * beam_far_count / neighbor_total_count;
			facade_far_count = 100 * facade_far_count / neighbor_total_count;
			roof_far_count = 100 * roof_far_count / neighbor_total_count;
			
			int descriptor = pillar_count * 1000000 + beam_count * 10000 + facade_count * 100 + roof_count; //the neighborhood discriptor (8 numbers)
			int descriptor_1 = pillar_close_count * 1000000 + beam_close_count * 10000 + facade_close_count * 100 + roof_close_count;
			int descriptor_2 = pillar_far_count * 1000000 + beam_far_count * 10000 + facade_far_count * 100 + roof_far_count;

			//TODO MULLS: fix later, keypoints would not be used in fine registration, so we do not need the timestamp (stored in curvature) and normal vector
			pt.curvature = descriptor;
			pt.normal[0] = descriptor_1;
			pt.normal[1] = descriptor_2;

			pt.intensity = accu_intensity / neighbor_total_count; //mean intensity of the nrighborhood
																	//pt.data_c[3] store the point curvature
																	//pt.data[3] store the height of the point above the ground

			// TODO MULLS: fix, use customed point type, you need a lot of porperties for saving linearity, planarity, curvature, semantic label and timestamp
			// However, within the template class, there might be a lot of problems (waiting for the code reproducing)

			cloud_out->points.push_back(pt);
		}
	}
	return true;
}

bool MullsFilter::non_max_suppress(pcl::PointCloud<MullsPoint>::Ptr &cloud_in_out, float non_max_radius,
						           bool kd_tree_already_built, const pcl::search::KdTree<MullsPoint>::Ptr &built_tree) {
	pcl::PointCloud<MullsPoint>::Ptr cloud_temp(new pcl::PointCloud<MullsPoint>());
	int pt_count_before = cloud_in_out->points.size();
	if (pt_count_before < 10)
		return false;

	std::sort(cloud_in_out->points.begin(), cloud_in_out->points.end(), [](const MullsPoint &a, const MullsPoint &b) { return a.data_c[3] > b.data_c[3]; });

	std::set<int, std::less<int>> unVisitedPtId;
	std::set<int, std::less<int>>::iterator iterUnseg;
	for (int i = 0; i < pt_count_before; ++i)
		unVisitedPtId.insert(i);

	pcl::search::KdTree<MullsPoint>::Ptr tree(new pcl::search::KdTree<MullsPoint>());
	if (kd_tree_already_built)
		tree = built_tree;
	else
		tree->setInputCloud(cloud_in_out);

	std::vector<int> search_indices;
	std::vector<float> distances;
	int keypointnum = 0;
	do {
		keypointnum++;
		std::vector<int>().swap(search_indices);
		std::vector<float>().swap(distances);

		int id;
		iterUnseg = unVisitedPtId.begin();
		id = *iterUnseg;
		cloud_temp->points.push_back(cloud_in_out->points[id]);
		unVisitedPtId.erase(id);

		tree->radiusSearch(cloud_in_out->points[id], non_max_radius, search_indices, distances);

		for (size_t i = 0u; i < search_indices.size(); i++)
			unVisitedPtId.erase(search_indices[i]);

	} while (!unVisitedPtId.empty());

	cloud_in_out->points.swap(cloud_temp->points);

	int pt_count_after_nms = cloud_in_out->points.size();
	return true;
}

bool MullsFilter::non_max_suppress(pcl::PointCloud<MullsPoint>::Ptr &cloud_in,
								   pcl::PointCloud<MullsPoint>::Ptr &cloud_out, float nms_radius,
								   bool distance_adaptive_on, float unit_dist,
								   bool kd_tree_already_built, const pcl::search::KdTree<MullsPoint>::Ptr &built_tree) {
	pcl::PointCloud<MullsPoint>::Ptr cloud_temp(new pcl::PointCloud<MullsPoint>());

	int pt_count_before = cloud_in->points.size();
	if (pt_count_before < 10)
		return false;

	std::sort(cloud_in->points.begin(), cloud_in->points.end(), [](const MullsPoint &a, const MullsPoint &b) { return a.data_c[3] > b.data_c[3]; }); //using the unused data_c[3] to represent what we want

	std::set<int, std::less<int>> unVisitedPtId;
	std::set<int, std::less<int>>::iterator iterUnseg;
	for (int i = 0; i < pt_count_before; ++i)
		unVisitedPtId.insert(i);

	pcl::search::KdTree<MullsPoint>::Ptr tree(new pcl::search::KdTree<MullsPoint>());
	if (kd_tree_already_built)
		tree = built_tree;
	else
		tree->setInputCloud(cloud_in);

	std::vector<int> search_indices;
	std::vector<float> distances;
	int keypointnum = 0;
	do {
		keypointnum++;
		std::vector<int>().swap(search_indices);
		std::vector<float>().swap(distances);

		int id;
		iterUnseg = unVisitedPtId.begin();
		id = *iterUnseg;
		cloud_out->points.push_back(cloud_in->points[id]);
		unVisitedPtId.erase(id);

		float non_max_radius = nms_radius;

		if (distance_adaptive_on) {
			double dist = std::sqrt(cloud_in->points[id].x * cloud_in->points[id].x +
									cloud_in->points[id].y * cloud_in->points[id].y +
									cloud_in->points[id].z * cloud_in->points[id].z);
			if (dist > unit_dist) {
				non_max_radius = std::sqrt(dist / unit_dist) * nms_radius;
				//neighborhood_k = (int)(unit_dist / dist * nearest_k));
			}
		}

		tree->radiusSearch(cloud_in->points[id], non_max_radius, search_indices, distances);

		for (size_t i = 0; i < search_indices.size(); i++)
			unVisitedPtId.erase(search_indices[i]);

	} while (!unVisitedPtId.empty());
	int pt_count_after_nms = cloud_out->points.size();
	return true;
}

bool MullsFilter::fast_ground_filter(const pcl::PointCloud<MullsPoint>::Ptr &cloud_in,
									 pcl::PointCloud<MullsPoint>::Ptr &cloud_ground,
									 pcl::PointCloud<MullsPoint>::Ptr &cloud_ground_down,
									 pcl::PointCloud<MullsPoint>::Ptr &cloud_unground,
									 pcl::PointCloud<MullsPoint>::Ptr &cloud_curb,
									 int min_grid_pt_num, float grid_resolution, float max_height_difference,
									 float neighbor_height_diff, float max_ground_height,
									 int ground_random_down_rate, int ground_random_down_down_rate, int nonground_random_down_rate, int reliable_neighbor_grid_num_thre,
									 int estimate_ground_normal_method, float normal_estimation_radius, //estimate_ground_normal_method, 0: directly use (0,0,1), 1: estimate normal in fix radius neighborhood , 2: estimate normal in k nearest neighborhood, 3: use ransac to estimate plane coeffs in a grid
									 int distance_weight_downsampling_method, float standard_distance,  //standard distance: the distance where the distance_weight is 1
									 bool fixed_num_downsampling, int down_ground_fixed_num,
									 bool detect_curb_or_not, float intensity_thre,
									 bool apply_grid_wise_outlier_filter, float outlier_std_scale)  {
	PrincipleComponentAnalysis pca_estimator;

	pcl::PointCloud<MullsPoint>::Ptr cloud_ground_full(new pcl::PointCloud<MullsPoint>());

	int reliable_grid_pts_count_thre = min_grid_pt_num - 1;
	int count_checkpoint = 0;
	float sum_height = 0.001;
	float appro_mean_height;
	float min_ground_height = max_ground_height;
	float underground_noise_thre = -FLT_MAX;
	float non_ground_height_thre;
	float distance_weight;

	//For some points,  calculating the approximate mean height
	for (size_t j = 0u; j < cloud_in->points.size(); j++) {
		if (j % 100 == 0) {
			sum_height += cloud_in->points[j].z;
			count_checkpoint++;
		}
	}
	appro_mean_height = sum_height / count_checkpoint;

	non_ground_height_thre = appro_mean_height + max_ground_height;
	//sometimes, there would be some underground ghost points (noise), however, these points would be removed by scanner filter
	//float underground_noise_thre = appro_mean_height - max_ground_height;  // this is a keyparameter.

	Bounds bounds;
	CenterPoint center_pt;
	get_cloud_bbx_cpt(cloud_in, bounds, center_pt); //Inherited from its parent class, use this->

	//Construct Grid
	int row, col, num_grid;
	row = ceil((bounds.max_y - bounds.min_y) / grid_resolution);
	col = ceil((bounds.max_x - bounds.min_x) / grid_resolution);
	num_grid = row * col;

	grid_t *grid = new grid_t[num_grid];

	//Each grid
	for (int i = 0; i < num_grid; i++) {
		grid[i].min_z = FLT_MAX;
		grid[i].neighbor_min_z = FLT_MAX;
	}

	//Each point ---> determine the grid to which the point belongs
	for (size_t j = 0u; j < cloud_in->points.size(); j++) {
		int temp_row, temp_col, temp_id;
		temp_col = floor((cloud_in->points[j].x - bounds.min_x) / grid_resolution);
		temp_row = floor((cloud_in->points[j].y - bounds.min_y) / grid_resolution);
		temp_id = temp_row * col + temp_col;
		if (temp_id >= 0 && temp_id < num_grid) {
			if (distance_weight_downsampling_method > 0 && !grid[temp_id].pts_count) {
				grid[temp_id].dist2station = std::sqrt(cloud_in->points[j].x * cloud_in->points[j].x + cloud_in->points[j].y * cloud_in->points[j].y + cloud_in->points[j].z * cloud_in->points[j].z);
			}

			if (cloud_in->points[j].z > non_ground_height_thre) {
				distance_weight = 1.0 * standard_distance / (grid[temp_id].dist2station + 0.0001); //avoiding Floating point exception
				int nonground_random_down_rate_temp = nonground_random_down_rate;
				if (distance_weight_downsampling_method == 1) //linear weight
					nonground_random_down_rate_temp = (int)(distance_weight * nonground_random_down_rate + 1);
				else if (distance_weight_downsampling_method == 2) //quadratic weight
					nonground_random_down_rate_temp = (int)(distance_weight * distance_weight * nonground_random_down_rate + 1);

				if (j % nonground_random_down_rate_temp == 0 || cloud_in->points[j].intensity > intensity_thre) {
					cloud_in->points[j].data[3] = cloud_in->points[j].z - (appro_mean_height - 3.0); //data[3] stores the approximate point height above ground
					cloud_unground->points.push_back(cloud_in->points[j]);
				}
			}
			else if (cloud_in->points[j].z > underground_noise_thre) {
				grid[temp_id].pts_count++;
				grid[temp_id].point_id.push_back(j);
				if (cloud_in->points[j].z < grid[temp_id].min_z) {
					grid[temp_id].min_z = cloud_in->points[j].z;
					grid[temp_id].neighbor_min_z = cloud_in->points[j].z;
				}
			}
		}
	}

	if (apply_grid_wise_outlier_filter)
	{
		//Each grid: Check outlier //calculate mean and standard deviation of z in one grid, then set mean-2*std as the threshold for outliers
		for (int i = 0; i < num_grid; i++) {
			if (grid[i].pts_count >= min_grid_pt_num) {
				double sum_z = 0, sum_z2 = 0, std_z = 0, mean_z = 0;
				for (size_t j = 0u; j < grid[i].point_id.size(); j++)
					sum_z += cloud_in->points[grid[i].point_id[j]].z;
				mean_z = sum_z / grid[i].pts_count;
				for (size_t j = 0u; j < grid[i].point_id.size(); j++)
					sum_z2 += (cloud_in->points[grid[i].point_id[j]].z - mean_z) * (cloud_in->points[grid[i].point_id[j]].z - mean_z);
				std_z = std::sqrt(sum_z2 / grid[i].pts_count);
				grid[i].min_z_outlier_thre = mean_z - outlier_std_scale * std_z;
				grid[i].min_z = std::max(grid[i].min_z, grid[i].min_z_outlier_thre);
				grid[i].neighbor_min_z = grid[i].min_z;
			}
		}
	}

	//Each grid
	for (int m = 0; m < num_grid; m++) {
		int temp_row, temp_col;
		temp_row = m / col;
		temp_col = m % col;
		if (temp_row >= 1 && temp_row <= row - 2 && temp_col >= 1 && temp_col <= col - 2) {
			for (int j = -1; j <= 1; j++) {
				for (int k = -1; k <= 1; k++) {
					grid[m].neighbor_min_z = std::min(grid[m].neighbor_min_z, grid[m + j * col + k].min_z);
					if (grid[m + j * col + k].pts_count > reliable_grid_pts_count_thre)
						grid[m].reliable_neighbor_grid_num++;
				}
			}
		}
	}

	double consuming_time_ransac = 0.0;

	std::vector<pcl::PointCloud<MullsPoint>::Ptr> grid_ground_pcs(num_grid);
	std::vector<pcl::PointCloud<MullsPoint>::Ptr> grid_unground_pcs(num_grid);
	for (int i = 0; i < num_grid; i++) {
		pcl::PointCloud<MullsPoint>::Ptr grid_ground_pc_temp(new pcl::PointCloud<MullsPoint>);
		grid_ground_pcs[i] = grid_ground_pc_temp;
		pcl::PointCloud<MullsPoint>::Ptr grid_unground_pc_temp(new pcl::PointCloud<MullsPoint>);
		grid_unground_pcs[i] = grid_unground_pc_temp;
	}

	//For each grid
	omp_set_num_threads(std::min(6, omp_get_max_threads()));
#pragma omp parallel for
	for (int i = 0; i < num_grid; i++) {
		pcl::PointCloud<MullsPoint>::Ptr grid_ground(new pcl::PointCloud<MullsPoint>);
		//Filtering some grids with too little points
		if (grid[i].pts_count >= min_grid_pt_num && grid[i].reliable_neighbor_grid_num >= reliable_neighbor_grid_num_thre) {
			int ground_random_down_rate_temp = ground_random_down_rate;
			int nonground_random_down_rate_temp = nonground_random_down_rate;
			distance_weight = 1.0 * standard_distance / (grid[i].dist2station + 0.0001);
			//linear weight
			if (distance_weight_downsampling_method == 1) {
				ground_random_down_rate_temp = (int)(distance_weight * ground_random_down_rate + 1);
				nonground_random_down_rate_temp = (int)(distance_weight * nonground_random_down_rate + 1);
			}
			else if (distance_weight_downsampling_method == 2) {//quadratic weight
				ground_random_down_rate_temp = (int)(distance_weight * distance_weight * ground_random_down_rate + 1);
				nonground_random_down_rate_temp = (int)(distance_weight * distance_weight * nonground_random_down_rate + 1);
			}
			if (grid[i].min_z - grid[i].neighbor_min_z < neighbor_height_diff) {
				for (size_t j = 0u; j < grid[i].point_id.size(); j++) {
					if (cloud_in->points[grid[i].point_id[j]].z > grid[i].min_z_outlier_thre) {
						if (cloud_in->points[grid[i].point_id[j]].z - grid[i].min_z < max_height_difference) {
							if (estimate_ground_normal_method == 3)
								grid_ground->points.push_back(cloud_in->points[grid[i].point_id[j]]);
							else {
								if (j % ground_random_down_rate_temp == 0) {
									if (estimate_ground_normal_method == 0) {
										cloud_in->points[grid[i].point_id[j]].normal_x = 0.0;
										cloud_in->points[grid[i].point_id[j]].normal_y = 0.0;
										cloud_in->points[grid[i].point_id[j]].normal_z = 1.0;
									}
									grid_ground_pcs[i]->points.push_back(cloud_in->points[grid[i].point_id[j]]);
								}
							}
						} else {// inner grid unground points
							if (j % nonground_random_down_rate_temp == 0 || cloud_in->points[grid[i].point_id[j]].intensity > intensity_thre) {//extract more points on signs and vehicle license plate
								cloud_in->points[grid[i].point_id[j]].data[3] = cloud_in->points[grid[i].point_id[j]].z - grid[i].min_z; //data[3] stores the point height above ground
								grid_unground_pcs[i]->points.push_back(cloud_in->points[grid[i].point_id[j]]);
							}
						}
					}
				}
			} else {//unground grid
				for (size_t j = 0u; j < grid[i].point_id.size(); j++) {
					if (cloud_in->points[grid[i].point_id[j]].z > grid[i].min_z_outlier_thre &&
						(j % nonground_random_down_rate_temp == 0 || cloud_in->points[grid[i].point_id[j]].intensity > intensity_thre)) {
						cloud_in->points[grid[i].point_id[j]].data[3] = cloud_in->points[grid[i].point_id[j]].z - grid[i].neighbor_min_z; //data[3] stores the point height above ground
						grid_unground_pcs[i]->points.push_back(cloud_in->points[grid[i].point_id[j]]);
					}
				}
			}
			if (estimate_ground_normal_method == 3 && grid_ground->points.size() >= size_t(min_grid_pt_num)) {
				float normal_x, normal_y, normal_z;

				//RANSAC iteration number equation: p=1-(1-r^N)^M,
				//r is the inlier ratio (> 0.75 in our case), N is 3 in our case (3 points can fit a plane), to get a confidence > 0.99, we need about 20 iteration (M=20)
				estimate_ground_normal_by_ransac(grid_ground, 0.3 * max_height_difference, 20, normal_x, normal_y, normal_z);

				for (size_t j = 0u; j < grid_ground->points.size(); j++)
				{
					if (j % ground_random_down_rate_temp == 0 && std::abs(normal_z) > 0.8) //53 deg
					{
						grid_ground->points[j].normal_x = normal_x;
						grid_ground->points[j].normal_y = normal_y;
						grid_ground->points[j].normal_z = normal_z;
						grid_ground_pcs[i]->points.push_back(grid_ground->points[j]); //Add to ground points
																						//cloud_ground->points.push_back(grid_ground->points[j]); //Add to ground points
					}
				}
			}
			pcl::PointCloud<MullsPoint>().swap(*grid_ground);
		}
	}

	//combine the ground and unground points
	for (int i = 0; i < num_grid; i++) {
		cloud_ground->points.insert(cloud_ground->points.end(), grid_ground_pcs[i]->points.begin(), grid_ground_pcs[i]->points.end());
		cloud_unground->points.insert(cloud_unground->points.end(), grid_unground_pcs[i]->points.begin(), grid_unground_pcs[i]->points.end());
	}

	//free memory
	delete[] grid;

	int normal_estimation_neighbor_k = 2 * min_grid_pt_num;
	pcl::PointCloud<pcl::Normal>::Ptr ground_normal(new pcl::PointCloud<pcl::Normal>);
	if (estimate_ground_normal_method == 1)
		pca_estimator.get_normal_pcar(cloud_ground, normal_estimation_radius, ground_normal);
	else if (estimate_ground_normal_method == 2)
		pca_estimator.get_normal_pcak(cloud_ground, normal_estimation_neighbor_k, ground_normal);

	for (size_t i = 0u; i < cloud_ground->points.size(); i++) {
		if (estimate_ground_normal_method == 1 || estimate_ground_normal_method == 2) {
			cloud_ground->points[i].normal_x = ground_normal->points[i].normal_x;
			cloud_ground->points[i].normal_y = ground_normal->points[i].normal_y;
			cloud_ground->points[i].normal_z = ground_normal->points[i].normal_z;
		}
		if (!fixed_num_downsampling) {
			//LOG(INFO)<<cloud_ground->points[i].normal_x << "," << cloud_ground->points[i].normal_y << "," << cloud_ground->points[i].normal_z;
			if (i % ground_random_down_down_rate == 0)
				cloud_ground_down->points.push_back(cloud_ground->points[i]);
		}
	}

	if (fixed_num_downsampling)
		random_downsample_pcl(cloud_ground, cloud_ground_down, down_ground_fixed_num);

	pcl::PointCloud<pcl::Normal>().swap(*ground_normal);
	return true;
}

bool MullsFilter::plane_seg_ransac(const pcl::PointCloud<MullsPoint>::Ptr &cloud,
								   float threshold, int max_iter, 
								   pcl::PointCloud<MullsPoint>::Ptr &planecloud, 
								   pcl::ModelCoefficients::Ptr &coefficients) {
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	// Create the segmentation object
	pcl::SACSegmentation<MullsPoint> sacseg;

	// Optional
	sacseg.setOptimizeCoefficients(true);

	// Mandatory
	sacseg.setModelType(pcl::SACMODEL_PLANE);
	sacseg.setMethodType(pcl::SAC_RANSAC);
	sacseg.setDistanceThreshold(threshold);
	sacseg.setMaxIterations(max_iter);

	sacseg.setInputCloud(cloud);
	sacseg.segment(*inliers, *coefficients);

	if (inliers->indices.size() == 0) {
		LOG(ERROR) << "MULLS Couldn't estimate a planar model.";
	}

	for (size_t i = 0; i < inliers->indices.size(); ++i) {
		planecloud->push_back(cloud->points[inliers->indices[i]]);
	}
	return true;
}

bool MullsFilter::estimate_ground_normal_by_ransac(pcl::PointCloud<MullsPoint>::Ptr &grid_ground,
										           float dist_thre, int max_iter, float &nx, float &ny, float &nz) {
	pcl::PointCloud<MullsPoint>::Ptr grid_ground_fit(new pcl::PointCloud<MullsPoint>);
	pcl::ModelCoefficients::Ptr grid_coeff(new pcl::ModelCoefficients);
	plane_seg_ransac(grid_ground, dist_thre, max_iter, grid_ground_fit, grid_coeff);

	grid_ground.swap(grid_ground_fit);
	nx = grid_coeff->values[0];
	ny = grid_coeff->values[1];
	nz = grid_coeff->values[2];
	return true;
}

bool MullsFilter::classify_nground_pts(pcl::PointCloud<MullsPoint>::Ptr &cloud_in,
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
									   bool fixed_num_downsampling, int pillar_down_fixed_num, int facade_down_fixed_num, int beam_down_fixed_num,
									   int roof_down_fixed_num, int unground_down_fixed_num,
									   float beam_height_max, float roof_height_min,
									   float feature_pts_ratio_guess, bool sharpen_with_nms,
									   bool use_distance_adaptive_pca) {
	if (fixed_num_downsampling)
		random_downsample_pcl(cloud_in, unground_down_fixed_num);

	//Do PCA
	PrincipleComponentAnalysis pca_estimator;
	std::vector<pca_feature_t> cloud_features;

	pcl::KdTreeFLANN<MullsPoint>::Ptr tree(new pcl::KdTreeFLANN<MullsPoint>);
	tree->setInputCloud(cloud_in);

	float unit_distance = 30.0;
	pca_estimator.get_pc_pca_feature(cloud_in, cloud_features, tree, neighbor_searching_radius, neighbor_k, 1, pca_down_rate, use_distance_adaptive_pca, unit_distance);

	//the radius should be larger for far away points
	std::vector<int> index_with_feature(cloud_in->points.size(), 0); // 0 - not special points, 1 - pillar, 2 - beam, 3 - facade, 4 - roof

	for (size_t i = 0u; i < cloud_in->points.size(); i++) {
		if (cloud_features[i].pt_num > neigh_k_min) {
			if (cloud_features[i].linear_2 > edge_thre) {
				if (std::abs(cloud_features[i].vectors.principalDirection.z()) > linear_vertical_sin_high_thre) {
					pca_estimator.assign_normal(cloud_in->points[i], cloud_features[i], false);
					cloud_pillar->points.push_back(cloud_in->points[i]);
					index_with_feature[i] = 1;
				} else if (std::abs(cloud_features[i].vectors.principalDirection.z()) < linear_vertical_sin_low_thre && cloud_in->points[i].z < beam_height_max) {
					pca_estimator.assign_normal(cloud_in->points[i], cloud_features[i], false);
					cloud_beam->points.push_back(cloud_in->points[i]);
					index_with_feature[i] = 2;
				} else {
					;
				}

				if (!sharpen_with_nms && cloud_features[i].linear_2 > edge_thre_down) {
					if (std::abs(cloud_features[i].vectors.principalDirection.z()) > linear_vertical_sin_high_thre)
						cloud_pillar_down->points.push_back(cloud_in->points[i]);
					else if (std::abs(cloud_features[i].vectors.principalDirection.z()) < linear_vertical_sin_low_thre && cloud_in->points[i].z < beam_height_max)
						cloud_beam_down->points.push_back(cloud_in->points[i]);
					else {
						;
					}
				}
			} else if (cloud_features[i].planar_2 > planar_thre) {
				if (std::abs(cloud_features[i].vectors.normalDirection.z()) > planar_vertical_sin_high_thre && cloud_in->points[i].z > roof_height_min) {
					pca_estimator.assign_normal(cloud_in->points[i], cloud_features[i], true);
					cloud_roof->points.push_back(cloud_in->points[i]);
					index_with_feature[i] = 4;
				} else if (std::abs(cloud_features[i].vectors.normalDirection.z()) < planar_vertical_sin_low_thre) {
					pca_estimator.assign_normal(cloud_in->points[i], cloud_features[i], true);
					cloud_facade->points.push_back(cloud_in->points[i]);
					index_with_feature[i] = 3;
				} else {
					;
				}
				if (!sharpen_with_nms && cloud_features[i].planar_2 > planar_thre_down) {
					if (std::abs(cloud_features[i].vectors.normalDirection.z()) > planar_vertical_sin_high_thre && cloud_in->points[i].z > roof_height_min)
						cloud_roof_down->points.push_back(cloud_in->points[i]);
					else if (std::abs(cloud_features[i].vectors.normalDirection.z()) < planar_vertical_sin_low_thre)
						cloud_facade_down->points.push_back(cloud_in->points[i]);
					else {
						;
					}
				}
			}
		}
	}

	//According to the parameter 'extract_vertex_points_method' (0,1,2...)
	if (curvature_thre < 1e-8) // set stablilty_thre as 0 to disable the vertex extraction
		extract_vertex_points_method = 0;

	//Find Edge points by picking high curvature points among the neighborhood of unground geometric feature points (2)
	if (extract_vertex_points_method == 2) {
		float vertex_feature_ratio_thre = feature_pts_ratio_guess / pca_down_rate;
		for (size_t i = 0u; i < cloud_in->points.size(); i++) {
			// if (index_with_feature[i] == 0)
			// 	cloud_vertex->points.push_back(cloud_in->points[i]);
			//curvature_thre means curvature_thre here
			if (index_with_feature[i] == 0 && cloud_features[i].pt_num > neigh_k_min && cloud_features[i].curvature > curvature_thre) {
				int geo_feature_point_count = 0;
				for (size_t j = 0u; j < cloud_features[i].neighbor_indices.size(); j++) {
					if (index_with_feature[cloud_features[i].neighbor_indices[j]])
						geo_feature_point_count++;
				}
				//most of the neighbors are feature points 
				if (1.0 * geo_feature_point_count / cloud_features[i].pt_num > vertex_feature_ratio_thre) {
					pca_estimator.assign_normal(cloud_in->points[i], cloud_features[i], false);
					cloud_in->points[i].data_c[3] = 5.0 * cloud_features[i].curvature; //save in the un-used data_c[3]  (PointNormal4D)
					if (std::abs(cloud_features[i].vectors.principalDirection.z()) > linear_vertical_sin_high_thre) {
						cloud_pillar->points.push_back(cloud_in->points[i]);
						index_with_feature[i] = 1;
					} else if (std::abs(cloud_features[i].vectors.principalDirection.z()) < linear_vertical_sin_low_thre && cloud_in->points[i].z < beam_height_max) {
						cloud_beam->points.push_back(cloud_in->points[i]);
						index_with_feature[i] = 2;
					}
				}
			}
		}
	}

	//if extract_vertex_points_method == 0 ---> do not extract vertex points (0)
	//extract neighborhood feature descriptor for pillar points
	//Find Vertex (Edge) points by picking points with maximum local curvature (1)
	//if (extract_vertex_points_method == 1) //Deprecated
	//detect_key_pts(cloud_in, cloud_features, index_with_feature,cloud_vertex, 4.0 * curvature_thre, vertex_curvature_non_max_radius, 0.5 * curvature_thre);
	int min_neighbor_feature_pts = (int)(feature_pts_ratio_guess / pca_down_rate * neighbor_k) - 1;

	//get the vertex keypoints and encode its neighborhood in a simple descriptor
	encode_stable_points(cloud_in, cloud_vertex, cloud_features, index_with_feature,
							0.3 * curvature_thre, min_neighbor_feature_pts, neigh_k_min); //encode the keypoints, we will get a simple descriptor of the putable keypoints

	//Non_max_suppression of the feature points //TODO: add already built-kd tree here
	if (sharpen_with_nms) {
		float nms_radius = 0.25 * neighbor_searching_radius;
#pragma omp parallel sections
		{
#pragma omp section
			{
				if (pillar_down_fixed_num > 0)
					non_max_suppress(cloud_pillar, cloud_pillar_down, nms_radius);
			}
#pragma omp section
			{
				if (facade_down_fixed_num > 0)
					non_max_suppress(cloud_facade, cloud_facade_down, nms_radius);
			}
#pragma omp section
			{
				if (beam_down_fixed_num > 0)
					non_max_suppress(cloud_beam, cloud_beam_down, nms_radius);

				if (roof_down_fixed_num > 0)
					non_max_suppress(cloud_roof, cloud_roof_down, nms_radius);
			}
		}
	}

	if (fixed_num_downsampling)
	{
		random_downsample_pcl(cloud_pillar_down, pillar_down_fixed_num);
		int sector_num = 4;
		xy_normal_balanced_downsample(cloud_facade_down, (int)(facade_down_fixed_num / sector_num), sector_num);

		xy_normal_balanced_downsample(cloud_beam_down, (int)(beam_down_fixed_num / sector_num), sector_num); // here the normal is the primary vector
																												//random_downsample_pcl(cloud_roof_down, 100);
		random_downsample_pcl(cloud_roof_down, roof_down_fixed_num);
	}

	//Free the memory
	std::vector<pca_feature_t>().swap(cloud_features);
	std::vector<int>().swap(index_with_feature);
	LOG(INFO) << "MULLS Pillar: [" << cloud_pillar->points.size() << " | " << cloud_pillar_down->points.size() << "] Beam: [" << cloud_beam->points.size() << " | " << cloud_beam_down->points.size() << "] Facade: [" << cloud_facade->points.size() << " | " << cloud_facade_down->points.size() << "] Roof: [" << cloud_roof->points.size() << " | "
				<< cloud_roof_down->points.size() << "] Vertex: [" << cloud_vertex->points.size() << "].";

	return true;
}


bool MullsFilter::extract_semantic_pts(CloudBlockPtr in_block, float gf_grid_resolution,
							           float gf_max_grid_height_diff, float gf_neighbor_height_diff, float gf_max_ground_height,
							           int &gf_down_rate_ground, int &gf_downsample_rate_nonground,
							           float pca_neighbor_radius, int pca_neighbor_k,
							           float edge_thre, float planar_thre, float curvature_thre,
							           float edge_thre_down, float planar_thre_down, bool use_distance_adaptive_pca,
							           int distance_inverse_sampling_method,
							           float standard_distance,
							           int estimate_ground_normal_method,
							           float normal_estimation_radius, bool apply_scanner_filter, bool extract_curb_or_not,
							           int extract_vertex_points_method,
							           int gf_grid_pt_num_thre, int gf_reliable_neighbor_grid_thre,
							           int gf_down_down_rate_ground, int pca_neighbor_k_min, int pca_down_rate,
							           float intensity_thre,
							           float linear_vertical_sin_high_thre, float linear_vertical_sin_low_thre,
							           float planar_vertical_sin_high_thre, float planar_vertical_sin_low_thre,
							           bool sharpen_with_nms_on, bool fixed_num_downsampling, int ground_down_fixed_num,
							           int pillar_down_fixed_num, int facade_down_fixed_num, int beam_down_fixed_num,
							           int roof_down_fixed_num, int unground_down_fixed_num, float beam_height_max , float roof_height_min,
							           float approx_scanner_height, float underground_thre, float feature_pts_ratio_guess,
							           bool semantic_assisted, bool apply_roi_filtering, float roi_min_y, float roi_max_y) {
	//pre-processing
	if (apply_scanner_filter) {
		float self_ring_radius = 1.75;
		float ghost_radius = 20.0;
		float z_min = -approx_scanner_height - 4.0;
		float z_min_min = -approx_scanner_height + underground_thre;
		// filter the point cloud of the back of the vehicle itself and the underground
		scanner_filter(in_block->pc_raw, self_ring_radius, ghost_radius, z_min, z_min_min);
	}
	in_block->pc_down = in_block->pc_raw;

	random_downsample(in_block->pc_down, in_block->pc_sketch, in_block->pc_down->points.size() / 1024 + 1);

	//filtered point cloud --> ground & non-ground point cloud
	fast_ground_filter(in_block->pc_down, in_block->pc_ground, in_block->pc_ground_down, in_block->pc_unground, in_block->pc_vertex,
						gf_grid_pt_num_thre, gf_grid_resolution, gf_max_grid_height_diff, gf_neighbor_height_diff,
						gf_max_ground_height, gf_down_rate_ground, gf_down_down_rate_ground,
						gf_downsample_rate_nonground, gf_reliable_neighbor_grid_thre, estimate_ground_normal_method, normal_estimation_radius,
						distance_inverse_sampling_method, standard_distance, fixed_num_downsampling, ground_down_fixed_num, extract_curb_or_not,
						intensity_thre, apply_scanner_filter);

	float vertex_curvature_non_max_r = 1.5 * pca_neighbor_radius;

	//non-ground points --> planar (facade, roof) & linear (pillar, beam) & spherical (vertex) points
	classify_nground_pts(in_block->pc_unground, in_block->pc_pillar, in_block->pc_beam,
							in_block->pc_facade, in_block->pc_roof,
							in_block->pc_pillar_down, in_block->pc_beam_down, in_block->pc_facade_down,
							in_block->pc_roof_down, in_block->pc_vertex,
							pca_neighbor_radius, pca_neighbor_k, pca_neighbor_k_min, pca_down_rate,
							edge_thre, planar_thre, edge_thre_down, planar_thre_down,
							extract_vertex_points_method, curvature_thre, vertex_curvature_non_max_r,
							linear_vertical_sin_high_thre, linear_vertical_sin_low_thre,
							planar_vertical_sin_high_thre, planar_vertical_sin_low_thre,
							fixed_num_downsampling, pillar_down_fixed_num, facade_down_fixed_num,
							beam_down_fixed_num, roof_down_fixed_num, unground_down_fixed_num,
							beam_height_max, roof_height_min, feature_pts_ratio_guess,
							sharpen_with_nms_on, use_distance_adaptive_pca);

	//using semantic mask predicted by neural network to refine the detected geometric feature points
	if (semantic_assisted) //Deprecated
		filter_with_semantic_mask(in_block); //currently disabled

	//transform the feature points back to the scanner's coordinate system
	//in_block->transform_feature(in_block->pose_lo.inverse(), true);

	in_block->down_feature_point_num = in_block->pc_ground_down->points.size() + in_block->pc_pillar_down->points.size() + in_block->pc_beam_down->points.size() +
										in_block->pc_facade_down->points.size() + in_block->pc_roof_down->points.size() + in_block->pc_vertex->points.size();

	return true;
}

//TODO: jingzhao use semantic filter the features
void MullsFilter::filter_with_semantic_mask(CloudBlockPtr in_block, const std::string mask_feature_type) {
	float labeled_radius = 45.0;
	//ground
	if (mask_feature_type[0] == '1') {
		pcl::PointCloud<MullsPoint>::Ptr cloud_temp(new pcl::PointCloud<MullsPoint>);
		for (size_t i = 0u; i < in_block->pc_ground->points.size(); i++) {
			int label = (int)(in_block->pc_ground->points[i].curvature);
			float dist2 = in_block->pc_ground->points[i].x * in_block->pc_ground->points[i].x + in_block->pc_ground->points[i].y * in_block->pc_ground->points[i].y;
			if (label == 40 || label == 44 || label == 48 || label == 49 || label == 72 || dist2 > labeled_radius * labeled_radius)
				cloud_temp->points.push_back(in_block->pc_ground->points[i]);
		}
		cloud_temp->points.swap(in_block->pc_ground->points);

		pcl::PointCloud<MullsPoint>::Ptr cloud_temp2(new pcl::PointCloud<MullsPoint>);
		for (size_t i = 0u; i < in_block->pc_ground_down->points.size(); i++) {
			int label = (int)(in_block->pc_ground_down->points[i].curvature);
			float dist2 = in_block->pc_ground_down->points[i].x * in_block->pc_ground_down->points[i].x + in_block->pc_ground_down->points[i].y * in_block->pc_ground_down->points[i].y;
			if (label == 40 || label == 44 || label == 48 || label == 49 || label == 72 || dist2 > labeled_radius * labeled_radius)
				cloud_temp2->points.push_back(in_block->pc_ground_down->points[i]);
		}
		cloud_temp2->points.swap(in_block->pc_ground_down->points);
	}

	//facade
	if (mask_feature_type[2] == '1') {
		pcl::PointCloud<MullsPoint>::Ptr cloud_temp3(new pcl::PointCloud<MullsPoint>);
		for (size_t i = 0u; i < in_block->pc_facade->points.size(); i++) {
			int label = (int)(in_block->pc_facade->points[i].curvature);
			float dist2 = in_block->pc_facade->points[i].x * in_block->pc_facade->points[i].x + in_block->pc_facade->points[i].y * in_block->pc_facade->points[i].y;
			if (label == 50 || label == 13 || label == 51 || dist2 > labeled_radius * labeled_radius)
				cloud_temp3->points.push_back(in_block->pc_facade->points[i]);
		}
		cloud_temp3->points.swap(in_block->pc_facade->points);

		pcl::PointCloud<MullsPoint>::Ptr cloud_temp4(new pcl::PointCloud<MullsPoint>);
		for (size_t i = 0u; i < in_block->pc_facade_down->points.size(); i++) {
			int label = (int)(in_block->pc_facade_down->points[i].curvature);
			float dist2 = in_block->pc_facade_down->points[i].x * in_block->pc_facade_down->points[i].x + in_block->pc_facade_down->points[i].y * in_block->pc_facade_down->points[i].y;
			if (label == 50 || label == 13 || label == 51 || dist2 > labeled_radius * labeled_radius)
				cloud_temp4->points.push_back(in_block->pc_facade_down->points[i]);
		}
		cloud_temp4->points.swap(in_block->pc_facade_down->points);
	}

	//pillar
	if (mask_feature_type[1] == '1') {
		pcl::PointCloud<MullsPoint>::Ptr cloud_temp5(new pcl::PointCloud<MullsPoint>);
		for (size_t i = 0u; i < in_block->pc_pillar->points.size(); i++)
		{
			int label = (int)(in_block->pc_pillar->points[i].curvature);
			float dist2 = in_block->pc_pillar->points[i].x * in_block->pc_pillar->points[i].x + in_block->pc_pillar->points[i].y * in_block->pc_pillar->points[i].y;
			if (label == 71 || label == 80 || label == 81 || dist2 > labeled_radius * labeled_radius)
				cloud_temp5->points.push_back(in_block->pc_pillar->points[i]);
		}
		cloud_temp5->points.swap(in_block->pc_pillar->points);

		pcl::PointCloud<MullsPoint>::Ptr cloud_temp6(new pcl::PointCloud<MullsPoint>);
		for (size_t i = 0u; i < in_block->pc_pillar_down->points.size(); i++)
		{
			int label = (int)(in_block->pc_pillar_down->points[i].curvature);
			float dist2 = in_block->pc_pillar_down->points[i].x * in_block->pc_pillar_down->points[i].x + in_block->pc_pillar_down->points[i].y * in_block->pc_pillar_down->points[i].y;
			if (label == 71 || label == 80 || label == 81 || dist2 > labeled_radius * labeled_radius)
				cloud_temp6->points.push_back(in_block->pc_pillar_down->points[i]);
		}
		cloud_temp6->points.swap(in_block->pc_pillar_down->points);
	}

	//beam
	if (mask_feature_type[3] == '1') {
		pcl::PointCloud<MullsPoint>::Ptr cloud_temp7(new pcl::PointCloud<MullsPoint>);
		for (size_t i = 0u; i < in_block->pc_beam->points.size(); i++) {
			int label = (int)(in_block->pc_beam->points[i].curvature);
			float dist2 = in_block->pc_beam->points[i].x * in_block->pc_beam->points[i].x + in_block->pc_beam->points[i].y * in_block->pc_beam->points[i].y;
			if (label == 50 || label == 51 || label == 10)
				cloud_temp7->points.push_back(in_block->pc_beam->points[i]);
		}
		cloud_temp7->points.swap(in_block->pc_beam->points);

		pcl::PointCloud<MullsPoint>::Ptr cloud_temp8(new pcl::PointCloud<MullsPoint>);
		for (size_t i = 0u; i < in_block->pc_beam_down->points.size(); i++) {
			int label = (int)(in_block->pc_beam_down->points[i].curvature);
			float dist2 = in_block->pc_beam_down->points[i].x * in_block->pc_beam_down->points[i].x + in_block->pc_beam_down->points[i].y * in_block->pc_beam_down->points[i].y;
			if (label == 50 || label == 51 || label == 10)
				cloud_temp8->points.push_back(in_block->pc_beam_down->points[i]);
		}
		cloud_temp8->points.swap(in_block->pc_beam_down->points);
	}

	LOG(INFO) << "Feature point number after semantic mask filtering: "
				<< "Ground: [" << in_block->pc_ground->points.size() << " | " << in_block->pc_ground_down->points.size()
				<< "] Pillar: [" << in_block->pc_pillar->points.size() << " | " << in_block->pc_pillar_down->points.size()
				<< "] Facade: [" << in_block->pc_facade->points.size() << " | " << in_block->pc_facade_down->points.size()
				<< "] Beam: [" << in_block->pc_beam->points.size() << " | " << in_block->pc_beam_down->points.size() << "]";
}

bool MullsFilter::get_cloud_pair_intersection(Bounds &intersection_bbx,
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
											  bool use_more_points) {
	bbx_filter(pc_ground_tc, intersection_bbx);
	bbx_filter(pc_pillar_tc, intersection_bbx);
	bbx_filter(pc_beam_tc, intersection_bbx);
	bbx_filter(pc_facade_tc, intersection_bbx);
	bbx_filter(pc_roof_tc, intersection_bbx);
	bbx_filter(pc_vertex_tc, intersection_bbx);
	if (use_more_points) {
		bbx_filter(pc_ground_sc, intersection_bbx);
		bbx_filter(pc_pillar_sc, intersection_bbx);
		bbx_filter(pc_beam_sc, intersection_bbx);
		bbx_filter(pc_facade_sc, intersection_bbx);
		bbx_filter(pc_roof_sc, intersection_bbx);
		bbx_filter(pc_vertex_sc, intersection_bbx);
	} else {
		bbx_filter(pc_ground_sc, intersection_bbx);
		bbx_filter(pc_pillar_sc, intersection_bbx);
		bbx_filter(pc_beam_sc, intersection_bbx);
		bbx_filter(pc_facade_sc, intersection_bbx);
		bbx_filter(pc_roof_sc, intersection_bbx);
		bbx_filter(pc_vertex_sc, intersection_bbx);
	}
	return true;
}

} // namespace common
} // namespace mapping_framework