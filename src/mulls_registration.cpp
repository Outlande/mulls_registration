#include <glog/logging.h>

#include "mulls_registration.h"

MullsRegistration::MullsRegistration(/* args */)
{}

MullsRegistration::~MullsRegistration()
{}

bool MullsRegistration::VoxelDownsample(const typename pcl::PointCloud<PointT>::Ptr& cloud_in, 
                                        typename pcl::PointCloud<PointT>::Ptr& cloud_out, float voxel_size) {
	//Set the downsampling_radius as 0 to disable the downsampling in order to save time for
	if (voxel_size < 0.001)
	{
		cloud_out = cloud_in;
		return false;
	}

	float inverse_voxel_size = 1.0f / voxel_size;
	Eigen::Vector4f min_p, max_p;
	pcl::getMinMax3D(*cloud_in, min_p, max_p);
	Eigen::Vector4f gap_p; //boundingbox gap;
	gap_p = max_p - min_p;

	unsigned long long max_vx = ceil(gap_p.coeff(0) * inverse_voxel_size) + 1;
	unsigned long long max_vy = ceil(gap_p.coeff(1) * inverse_voxel_size) + 1;
	unsigned long long max_vz = ceil(gap_p.coeff(2) * inverse_voxel_size) + 1;

	if (max_vx * max_vy * max_vz >= std::numeric_limits<unsigned long long>::max())
	{
		LOG(WARNING) << "Filtering Failed: The number of box exceed the limit.";
		return 0;
	}

	unsigned long long mul_vx = max_vy * max_vz;
	unsigned long long mul_vy = max_vz;
	unsigned long long mul_vz = 1;

	std::vector<idpair_t> id_pairs(cloud_in->points.size());

	int i;
#pragma omp parallel for private(i) //Multi-thread
	for (i = 0; i < cloud_in->points.size(); i++)
	{
		unsigned long long vx = floor((cloud_in->points[i].x - min_p.coeff(0)) * inverse_voxel_size);
		unsigned long long vy = floor((cloud_in->points[i].y - min_p.coeff(1)) * inverse_voxel_size);
		unsigned long long vz = floor((cloud_in->points[i].z - min_p.coeff(2)) * inverse_voxel_size);

		unsigned long long voxel_idx = vx * mul_vx + vy * mul_vy + vz * mul_vz;
		idpair_t pair;
		pair.idx = i;
		pair.voxel_idx = voxel_idx;
		//id_pairs.push_back(pair);
		id_pairs[i] = pair;
	}

	//Do sorting
	std::sort(id_pairs.begin(), id_pairs.end());
	int begin_id = 0;
	while (begin_id < id_pairs.size())
	{
		cloud_out->emplace_back(cloud_in->points[id_pairs[begin_id].idx]);

		int compare_id = begin_id + 1;
		while (compare_id < id_pairs.size() && id_pairs[begin_id].voxel_idx == id_pairs[compare_id].voxel_idx)
			compare_id++;
		begin_id = compare_id;
	}
	//free the memory
	std::vector<idpair_t>().swap(id_pairs);
	return true;
}

bool MullsRegistration::random_downsample(const typename pcl::PointCloud<PointT>::Ptr &cloud_in,
						                  typename pcl::PointCloud<PointT>::Ptr &cloud_out, int downsample_ratio) {
	if (downsample_ratio > 1) {
		cloud_out->points.clear();
		for (int i = 0; i < cloud_in->points.size(); i++) {
			if (i % downsample_ratio == 0) {
				cloud_out->points.push_back(cloud_in->points[i]);
			}
		}
		return 1;
	}
	else
		return 0;
}


bool MullsRegistration::estimate_ground_normal_by_ransac(typename pcl::PointCloud<PointT>::Ptr &grid_ground,
										  float dist_thre, int max_iter, float &nx, float &ny, float &nz)
{
	CProceesing<PointT> cpro;

	typename pcl::PointCloud<PointT>::Ptr grid_ground_fit(new pcl::PointCloud<PointT>);
	pcl::ModelCoefficients::Ptr grid_coeff(new pcl::ModelCoefficients);
	cpro.plane_seg_ransac(grid_ground, dist_thre, max_iter, grid_ground_fit, grid_coeff);

	grid_ground.swap(grid_ground_fit);
	nx = grid_coeff->values[0];
	ny = grid_coeff->values[1];
	nz = grid_coeff->values[2];

	//LOG(INFO) << nx << "," << ny << "," << nz;
	return 1;
}


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
bool MullsRegistration::fast_ground_filter(const typename pcl::PointCloud<PointT>::Ptr &cloud_in,
						typename pcl::PointCloud<PointT>::Ptr &cloud_ground,
						typename pcl::PointCloud<PointT>::Ptr &cloud_ground_down,
						typename pcl::PointCloud<PointT>::Ptr &cloud_unground,
						typename pcl::PointCloud<PointT>::Ptr &cloud_curb,
						int min_grid_pt_num, float grid_resolution, float max_height_difference,
						float neighbor_height_diff, float max_ground_height,
						int ground_random_down_rate, int ground_random_down_down_rate, int nonground_random_down_rate, int reliable_neighbor_grid_num_thre,
						int estimate_ground_normal_method, float normal_estimation_radius, //estimate_ground_normal_method, 0: directly use (0,0,1), 1: estimate normal in fix radius neighborhood , 2: estimate normal in k nearest neighborhood, 3: use ransac to estimate plane coeffs in a grid
						int distance_weight_downsampling_method, float standard_distance,  //standard distance: the distance where the distance_weight is 1
						bool fixed_num_downsampling = false, int down_ground_fixed_num = 1000,
						bool detect_curb_or_not = false, float intensity_thre = FLT_MAX,
						bool apply_grid_wise_outlier_filter = false, float outlier_std_scale = 3.0) //current intensity_thre is for kitti dataset (TODO: disable it)
{
	PrincipleComponentAnalysis<PointT> pca_estimator;
	typename pcl::PointCloud<PointT>::Ptr cloud_ground_full(new pcl::PointCloud<PointT>());
	int reliable_grid_pts_count_thre = min_grid_pt_num - 1;
	int count_checkpoint = 0;
	float sum_height = 0.001;
	float appro_mean_height;
	float min_ground_height = max_ground_height;
	float underground_noise_thre = -FLT_MAX;
	float non_ground_height_thre;
	float distance_weight;

	//For some points,  calculating the approximate mean height
	for (int j = 0; j < cloud_in->points.size(); j++)
	{
		if (j % 100 == 0)
		{
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
	for (int i = 0; i < num_grid; i++)
	{
		grid[i].min_z = FLT_MAX;
		grid[i].neighbor_min_z = FLT_MAX;
	}

	//Each point ---> determine the grid to which the point belongs
	for (int j = 0; j < cloud_in->points.size(); j++)
	{
		int temp_row, temp_col, temp_id;
		temp_col = floor((cloud_in->points[j].x - bounds.min_x) / grid_resolution);
		temp_row = floor((cloud_in->points[j].y - bounds.min_y) / grid_resolution);
		temp_id = temp_row * col + temp_col;
		if (temp_id >= 0 && temp_id < num_grid)
		{
			if (distance_weight_downsampling_method > 0 && !grid[temp_id].pts_count)
			{
				grid[temp_id].dist2station = std::sqrt(cloud_in->points[j].x * cloud_in->points[j].x + cloud_in->points[j].y * cloud_in->points[j].y + cloud_in->points[j].z * cloud_in->points[j].z);
			}

			if (cloud_in->points[j].z > non_ground_height_thre)
			{
				distance_weight = 1.0 * standard_distance / (grid[temp_id].dist2station + 0.0001); //avoiding Floating point exception
				int nonground_random_down_rate_temp = nonground_random_down_rate;
				if (distance_weight_downsampling_method == 1) //linear weight
					nonground_random_down_rate_temp = (int)(distance_weight * nonground_random_down_rate + 1);
				else if (distance_weight_downsampling_method == 2) //quadratic weight
					nonground_random_down_rate_temp = (int)(distance_weight * distance_weight * nonground_random_down_rate + 1);

				if (j % nonground_random_down_rate_temp == 0 || cloud_in->points[j].intensity > intensity_thre)
				{
					cloud_in->points[j].data[3] = cloud_in->points[j].z - (appro_mean_height - 3.0); //data[3] stores the approximate point height above ground
					cloud_unground->points.push_back(cloud_in->points[j]);
				}
			}
			else if (cloud_in->points[j].z > underground_noise_thre)
			{
				grid[temp_id].pts_count++;
				grid[temp_id].point_id.push_back(j);
				if (cloud_in->points[j].z < grid[temp_id].min_z) //
				{
					grid[temp_id].min_z = cloud_in->points[j].z;
					grid[temp_id].neighbor_min_z = cloud_in->points[j].z;
				}
			}
		}
	}

	if (apply_grid_wise_outlier_filter)
	{
		//Each grid: Check outlier //calculate mean and standard deviation of z in one grid, then set mean-2*std as the threshold for outliers
		for (int i = 0; i < num_grid; i++)
		{
			if (grid[i].pts_count >= min_grid_pt_num)
			{
				double sum_z = 0, sum_z2 = 0, std_z = 0, mean_z = 0;
				for (int j = 0; j < grid[i].point_id.size(); j++)
					sum_z += cloud_in->points[grid[i].point_id[j]].z;
				mean_z = sum_z / grid[i].pts_count;
				for (int j = 0; j < grid[i].point_id.size(); j++)
					sum_z2 += (cloud_in->points[grid[i].point_id[j]].z - mean_z) * (cloud_in->points[grid[i].point_id[j]].z - mean_z);
				std_z = std::sqrt(sum_z2 / grid[i].pts_count);
				grid[i].min_z_outlier_thre = mean_z - outlier_std_scale * std_z;
				grid[i].min_z = std::max(grid[i].min_z, grid[i].min_z_outlier_thre);
				grid[i].neighbor_min_z = grid[i].min_z;
			}
		}
	}

	//Each grid
	for (int m = 0; m < num_grid; m++)
	{
		int temp_row, temp_col;
		temp_row = m / col;
		temp_col = m % col;
		if (temp_row >= 1 && temp_row <= row - 2 && temp_col >= 1 && temp_col <= col - 2)
		{
			for (int j = -1; j <= 1; j++) //row
			{
				for (int k = -1; k <= 1; k++) //col
				{
					grid[m].neighbor_min_z = std::min(grid[m].neighbor_min_z, grid[m + j * col + k].min_z);
					if (grid[m + j * col + k].pts_count > reliable_grid_pts_count_thre)
						grid[m].reliable_neighbor_grid_num++;
				}
			}
		}
	}

	std::vector<typename pcl::PointCloud<PointT>::Ptr> grid_ground_pcs(num_grid);
	std::vector<typename pcl::PointCloud<PointT>::Ptr> grid_unground_pcs(num_grid);
	for (int i = 0; i < num_grid; i++)
	{
		typename pcl::PointCloud<PointT>::Ptr grid_ground_pc_temp(new pcl::PointCloud<PointT>);
		grid_ground_pcs[i] = grid_ground_pc_temp;
		typename pcl::PointCloud<PointT>::Ptr grid_unground_pc_temp(new pcl::PointCloud<PointT>);
		grid_unground_pcs[i] = grid_unground_pc_temp;
	}

	//For each grid
	omp_set_num_threads(std::min(6, omp_get_max_threads()));
#pragma omp parallel for
	for (int i = 0; i < num_grid; i++)
	{
		typename pcl::PointCloud<PointT>::Ptr grid_ground(new pcl::PointCloud<PointT>);
		//Filtering some grids with too little points
		if (grid[i].pts_count >= min_grid_pt_num && grid[i].reliable_neighbor_grid_num >= reliable_neighbor_grid_num_thre)
		{
			int ground_random_down_rate_temp = ground_random_down_rate;
			int nonground_random_down_rate_temp = nonground_random_down_rate;
			distance_weight = 1.0 * standard_distance / (grid[i].dist2station + 0.0001);
			if (distance_weight_downsampling_method == 1) //linear weight
			{
				ground_random_down_rate_temp = (int)(distance_weight * ground_random_down_rate + 1);
				nonground_random_down_rate_temp = (int)(distance_weight * nonground_random_down_rate + 1);
			}
			else if (distance_weight_downsampling_method == 2) //quadratic weight
			{
				ground_random_down_rate_temp = (int)(distance_weight * distance_weight * ground_random_down_rate + 1);
				nonground_random_down_rate_temp = (int)(distance_weight * distance_weight * nonground_random_down_rate + 1);
			}
			//LOG(WARNING) << ground_random_down_rate_temp << "," << nonground_random_down_rate_temp;
			if (grid[i].min_z - grid[i].neighbor_min_z < neighbor_height_diff)
			{
				for (int j = 0; j < grid[i].point_id.size(); j++)
				{
					if (cloud_in->points[grid[i].point_id[j]].z > grid[i].min_z_outlier_thre)
					{
						if (cloud_in->points[grid[i].point_id[j]].z - grid[i].min_z < max_height_difference)
						{
							//cloud_ground_full->points.push_back(cloud_in->points[grid[i].point_id[j]]);
							if (estimate_ground_normal_method == 3)
								grid_ground->points.push_back(cloud_in->points[grid[i].point_id[j]]);
							else
							{
								if (j % ground_random_down_rate_temp == 0) // for example 10
								{
									if (estimate_ground_normal_method == 0)
									{
										cloud_in->points[grid[i].point_id[j]].normal_x = 0.0;
										cloud_in->points[grid[i].point_id[j]].normal_y = 0.0;
										cloud_in->points[grid[i].point_id[j]].normal_z = 1.0;
									}
									grid_ground_pcs[i]->points.push_back(cloud_in->points[grid[i].point_id[j]]);
									//cloud_ground->points.push_back(cloud_in->points[grid[i].point_id[j]]); //Add to ground points
								}
							}
						}
						else // inner grid unground points
						{
							if (j % nonground_random_down_rate_temp == 0 || cloud_in->points[grid[i].point_id[j]].intensity > intensity_thre) //extract more points on signs and vehicle license plate
							{
								cloud_in->points[grid[i].point_id[j]].data[3] = cloud_in->points[grid[i].point_id[j]].z - grid[i].min_z; //data[3] stores the point height above ground
								grid_unground_pcs[i]->points.push_back(cloud_in->points[grid[i].point_id[j]]);
								//cloud_unground->points.push_back(cloud_in->points[grid[i].point_id[j]]); //Add to nonground points
							}
						}
					}
				}
			}
			else //unground grid
			{
				for (int j = 0; j < grid[i].point_id.size(); j++)
				{
					if (cloud_in->points[grid[i].point_id[j]].z > grid[i].min_z_outlier_thre &&
						(j % nonground_random_down_rate_temp == 0 || cloud_in->points[grid[i].point_id[j]].intensity > intensity_thre))
					{
						cloud_in->points[grid[i].point_id[j]].data[3] = cloud_in->points[grid[i].point_id[j]].z - grid[i].neighbor_min_z; //data[3] stores the point height above ground
						grid_unground_pcs[i]->points.push_back(cloud_in->points[grid[i].point_id[j]]);
						//cloud_unground->points.push_back(cloud_in->points[grid[i].point_id[j]]); //Add to nonground points
					}
				}
			}
			if (estimate_ground_normal_method == 3 && grid_ground->points.size() >= min_grid_pt_num)
			{
				float normal_x, normal_y, normal_z;
				//RANSAC iteration number equation: p=1-(1-r^N)^M,
				//r is the inlier ratio (> 0.75 in our case), N is 3 in our case (3 points can fit a plane), to get a confidence > 0.99, we need about 20 iteration (M=20)
				estimate_ground_normal_by_ransac(grid_ground, 0.3 * max_height_difference, 20, normal_x, normal_y, normal_z);

				for (int j = 0; j < grid_ground->points.size(); j++)
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
				consuming_time_ransac += ground_ransac_time_per_grid.count() * 1000.0; //unit: ms
			}
			pcl::PointCloud<PointT>().swap(*grid_ground);
		}
	}

	//combine the ground and unground points
	for (int i = 0; i < num_grid; i++)
	{
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

	for (int i = 0; i < cloud_ground->points.size(); i++)
	{
		if (estimate_ground_normal_method == 1 || estimate_ground_normal_method == 2)
		{
			cloud_ground->points[i].normal_x = ground_normal->points[i].normal_x;
			cloud_ground->points[i].normal_y = ground_normal->points[i].normal_y;
			cloud_ground->points[i].normal_z = ground_normal->points[i].normal_z;
		}
		if (!fixed_num_downsampling)
		{
			if (i % ground_random_down_down_rate == 0)
				cloud_ground_down->points.push_back(cloud_ground->points[i]);
		}
	}

	if (fixed_num_downsampling)
		random_downsample_pcl(cloud_ground, cloud_ground_down, down_ground_fixed_num);

	pcl::PointCloud<pcl::Normal>().swap(*ground_normal);

	LOG(INFO) << "Ground: [" << cloud_ground->points.size() << " | " << cloud_ground_down->points.size() << "] Unground: [" << cloud_unground->points.size() << "].";
	return true;
}


//fixed number random downsampling
//when keep_number == 0, the output point cloud would be empty
bool MullsRegistration::random_downsample_pcl(typename pcl::PointCloud<PointT>::Ptr &cloud_in,
							typename pcl::PointCloud<PointT>::Ptr &cloud_out, int keep_number)
{
	if (cloud_in->points.size() <= keep_number)
	{
		cloud_out = cloud_in;
		return false;
	}
	else
	{
		if (keep_number == 0)
			return false;
		else
		{
			pcl::RandomSample<PointT> ran_sample(true); // Extract removed indices
			ran_sample.setInputCloud(cloud_in);
			ran_sample.setSample(keep_number);
			ran_sample.filter(*cloud_out);
			return true;
		}
	}
}

bool MullsRegistration::classify_nground_pts(typename pcl::PointCloud<PointT>::Ptr &cloud_in,
							  typename pcl::PointCloud<PointT>::Ptr &cloud_pillar,
							  typename pcl::PointCloud<PointT>::Ptr &cloud_beam,
							  typename pcl::PointCloud<PointT>::Ptr &cloud_facade,
							  typename pcl::PointCloud<PointT>::Ptr &cloud_roof,
							  typename pcl::PointCloud<PointT>::Ptr &cloud_pillar_down,
							  typename pcl::PointCloud<PointT>::Ptr &cloud_beam_down,
							  typename pcl::PointCloud<PointT>::Ptr &cloud_facade_down,
							  typename pcl::PointCloud<PointT>::Ptr &cloud_roof_down,
							  typename pcl::PointCloud<PointT>::Ptr &cloud_vertex,
							  float neighbor_searching_radius, int neighbor_k, int neigh_k_min, int pca_down_rate, // one in ${pca_down_rate} unground points would be select as the query points for calculating pca, the else would only be used as neighborhood points
							  float edge_thre, float planar_thre, float edge_thre_down, float planar_thre_down,
							  int extract_vertex_points_method, float curvature_thre, float vertex_curvature_non_max_radius,
							  float linear_vertical_sin_high_thre, float linear_vertical_sin_low_thre,
							  float planar_vertical_sin_high_thre, float planar_vertical_sin_low_thre,
							  bool fixed_num_downsampling = false, int pillar_down_fixed_num = 200, int facade_down_fixed_num = 800, int beam_down_fixed_num = 200,
							  int roof_down_fixed_num = 100, int unground_down_fixed_num = 20000,
							  float beam_height_max = FLT_MAX, float roof_height_min = -FLT_MAX,
							  float feature_pts_ratio_guess = 0.3, bool sharpen_with_nms = true,
							  bool use_distance_adaptive_pca = false)
{
	if (fixed_num_downsampling)
		random_downsample_pcl(cloud_in, unground_down_fixed_num);

	//Do PCA
	PrincipleComponentAnalysis<PointT> pca_estimator;
	std::vector<pca_feature_t> cloud_features;
	typename pcl::KdTreeFLANN<PointT>::Ptr tree(new pcl::KdTreeFLANN<PointT>);
	tree->setInputCloud(cloud_in);
	float unit_distance = 30.0;
	pca_estimator.get_pc_pca_feature(cloud_in, cloud_features, tree, neighbor_searching_radius, neighbor_k, 1, pca_down_rate, use_distance_adaptive_pca, unit_distance);

	//the radius should be larger for far away points
	std::vector<int> index_with_feature(cloud_in->points.size(), 0); // 0 - not special points, 1 - pillar, 2 - beam, 3 - facade, 4 - roof

	for (int i = 0; i < cloud_in->points.size(); i++)
	{
		if (cloud_features[i].pt_num > neigh_k_min)
		{
			if (cloud_features[i].linear_2 > edge_thre)
			{
				if (std::abs(cloud_features[i].vectors.principalDirection.z()) > linear_vertical_sin_high_thre)
				{
					pca_estimator.assign_normal(cloud_in->points[i], cloud_features[i], false);
					cloud_pillar->points.push_back(cloud_in->points[i]);
					index_with_feature[i] = 1;
				}
				else if (std::abs(cloud_features[i].vectors.principalDirection.z()) < linear_vertical_sin_low_thre && cloud_in->points[i].z < beam_height_max)
				{
					pca_estimator.assign_normal(cloud_in->points[i], cloud_features[i], false);
					cloud_beam->points.push_back(cloud_in->points[i]);
					index_with_feature[i] = 2;
				}
				else
				{
					;
				}

				if (!sharpen_with_nms && cloud_features[i].linear_2 > edge_thre_down)
				{
					if (std::abs(cloud_features[i].vectors.principalDirection.z()) > linear_vertical_sin_high_thre)
						cloud_pillar_down->points.push_back(cloud_in->points[i]);
					else if (std::abs(cloud_features[i].vectors.principalDirection.z()) < linear_vertical_sin_low_thre && cloud_in->points[i].z < beam_height_max)
						cloud_beam_down->points.push_back(cloud_in->points[i]);
					else
					{
						;
					}
				}
			}

			else if (cloud_features[i].planar_2 > planar_thre)
			{
				if (std::abs(cloud_features[i].vectors.normalDirection.z()) > planar_vertical_sin_high_thre && cloud_in->points[i].z > roof_height_min)
				{
					pca_estimator.assign_normal(cloud_in->points[i], cloud_features[i], true);
					cloud_roof->points.push_back(cloud_in->points[i]);
					index_with_feature[i] = 4;
				}
				else if (std::abs(cloud_features[i].vectors.normalDirection.z()) < planar_vertical_sin_low_thre)
				{
					pca_estimator.assign_normal(cloud_in->points[i], cloud_features[i], true);
					cloud_facade->points.push_back(cloud_in->points[i]);
					index_with_feature[i] = 3;
				}
				else
				{
					;
				}
				if (!sharpen_with_nms && cloud_features[i].planar_2 > planar_thre_down)
				{
					if (std::abs(cloud_features[i].vectors.normalDirection.z()) > planar_vertical_sin_high_thre && cloud_in->points[i].z > roof_height_min)
						cloud_roof_down->points.push_back(cloud_in->points[i]);
					else if (std::abs(cloud_features[i].vectors.normalDirection.z()) < planar_vertical_sin_low_thre)
						cloud_facade_down->points.push_back(cloud_in->points[i]);
					else
					{
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
	if (extract_vertex_points_method == 2)
	{
		float vertex_feature_ratio_thre = feature_pts_ratio_guess / pca_down_rate;
		for (int i = 0; i < cloud_in->points.size(); i++)
		{
			if (index_with_feature[i] == 0 && cloud_features[i].pt_num > neigh_k_min && cloud_features[i].curvature > curvature_thre) //curvature_thre means curvature_thre here
			{
				int geo_feature_point_count = 0;
				for (int j = 0; j < cloud_features[i].neighbor_indices.size(); j++)
				{
					if (index_with_feature[cloud_features[i].neighbor_indices[j]])
						geo_feature_point_count++;
				}
				if (1.0 * geo_feature_point_count / cloud_features[i].pt_num > vertex_feature_ratio_thre) //most of the neighbors are feature points
				{
					pca_estimator.assign_normal(cloud_in->points[i], cloud_features[i], false);
					cloud_in->points[i].normal[3] = 5.0 * cloud_features[i].curvature; //save in the un-used normal[3]  (PointNormal4D)
					if (std::abs(cloud_features[i].vectors.principalDirection.z()) > linear_vertical_sin_high_thre)
					{
						cloud_pillar->points.push_back(cloud_in->points[i]);
						index_with_feature[i] = 1;
					}
					else if (std::abs(cloud_features[i].vectors.principalDirection.z()) < linear_vertical_sin_low_thre && cloud_in->points[i].z < beam_height_max)
					{
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

	//LOG(WARNING)<< "encode ncc feature descriptor done";

	//Non_max_suppression of the feature points
	if (sharpen_with_nms)
	{
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
	LOG(INFO) << "Pillar: [" << cloud_pillar->points.size() << " | " << cloud_pillar_down->points.size() << "] Beam: [" << cloud_beam->points.size() << " | " << cloud_beam_down->points.size() << "] Facade: [" << cloud_facade->points.size() << " | " << cloud_facade_down->points.size() << "] Roof: [" << cloud_roof->points.size() << " | "
				<< cloud_roof_down->points.size() << "] Vertex: [" << cloud_vertex->points.size() << "].";

	return 1;
}

bool MullsRegistration::xy_normal_balanced_downsample(typename pcl::PointCloud<PointT>::Ptr &cloud_in_out,
									   				  int keep_number_per_sector, int sector_num)
{
	if (cloud_in_out->points.size() <= keep_number_per_sector)
		return false;

	typename pcl::PointCloud<PointT>::Ptr cloud_temp(new pcl::PointCloud<PointT>);

	//divide into ${sector_num} sectors according to normal_x and normal_y
	std::vector<typename pcl::PointCloud<PointT>::Ptr> sectors(sector_num);

	//initialization
	for (int j = 0; j < sector_num; j++)
		sectors[j].reset(new pcl::PointCloud<PointT>());

	//LOG(INFO) << "initialization done\n";

	double angle_per_sector = 360.0 / sector_num;

	for (int i = 0; i < cloud_in_out->points.size(); i++)
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
	//LOG(INFO) << "assign sector done\n";

	for (int j = 0; j < sector_num; j++)
	{
		random_downsample_pcl(sectors[j], keep_number_per_sector);
		cloud_temp->points.insert(cloud_temp->points.end(), sectors[j]->points.begin(), sectors[j]->points.end());

		//LOG(INFO) << "sector " << j << " ok.";
	}

	cloud_temp->points.swap(cloud_in_out->points);

	std::vector<typename pcl::PointCloud<PointT>::Ptr>().swap(sectors);

	return true;
}


//hard-coded (not a good way) to adjust the parameters on fly
void MullsRegistration::update_parameters_self_adaptive(int &gf_down_rate_ground, int &gf_downsample_rate_nonground,
										float &pca_neighbor_k,
										float &edge_thre, float &planar_thre,
										float &edge_thre_down, float &planar_thre_down,
										int ground_down_num, int facade_down_num, int pillar_down_num, int beam_down_num,
										int ground_down_num_min_expected = 500, int ground_down_num_max_expected = 1200,
										int non_ground_num_min_expected = 200, int non_ground_num_max_expected = 1600,
										int facade_down_num_expected = 400, int pillar_down_num_expected = 200, int beam_down_num_expected = 300)
{
	int non_ground_feature_num = facade_down_num + pillar_down_num;

	if (non_ground_feature_num < non_ground_num_min_expected) {
		gf_downsample_rate_nonground = std::max(1, gf_downsample_rate_nonground - non_ground_num_min_expected / non_ground_feature_num);
	}
}


bool MullsRegistration::ExtractSemanticPts(CloudBlockPtr in_block,
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
							  bool semantic_assisted = false, bool apply_roi_filtering = false, float roi_min_y = 0.0, float roi_max_y = 0.0)
{
	//pre-processing
	voxel_downsample(in_block->pc_raw, in_block->pc_down, vf_downsample_resolution);
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

	//TODO:(jingzhao.tyx): using semantic mask predicted by neural network to refine the detected geometric feature points
	// filter_with_semantic_mask(in_block);

	in_block->down_feature_point_num = in_block->pc_ground_down->points.size() + in_block->pc_pillar_down->points.size() + in_block->pc_beam_down->points.size() +
										in_block->pc_facade_down->points.size() + in_block->pc_roof_down->points.size() + in_block->pc_vertex->points.size();

	//update the parameters according to the situation
	if (use_adpative_parameters)
		update_parameters_self_adaptive(gf_down_rate_ground, gf_downsample_rate_nonground, pca_neighbor_radius,
										edge_thre, planar_thre, edge_thre_down, planar_thre_down,
										in_block->pc_ground_down->points.size(), in_block->pc_facade_down->points.size(),
										in_block->pc_pillar_down->points.size(), in_block->pc_beam_down->points.size());

    return true;
}