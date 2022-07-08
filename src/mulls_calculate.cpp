#include "mulls_calculate.h"

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
void MullsCalculate::compute_fpfh_feature(const pcl::PointCloud<MullsPoint>::Ptr &input_cloud,
							                 fpfhPtr &cloud_fpfh, float search_radius) {
	// Calculate the Point Normal
	// Estimate FPFH Feature
	pcl::FPFHEstimationOMP<MullsPoint, MullsPoint, pcl::FPFHSignature33> est_fpfh;
	est_fpfh.setNumberOfThreads(4);
	est_fpfh.setInputCloud(input_cloud);
	est_fpfh.setInputNormals(input_cloud);
	pcl::search::KdTree<MullsPoint>::Ptr tree(new pcl::search::KdTree<MullsPoint>());
	est_fpfh.setSearchMethod(tree);
	//est_fpfh.setKSearch(20);
	est_fpfh.setRadiusSearch(2.0 * search_radius);
	est_fpfh.compute(*cloud_fpfh);
}

double MullsCalculate::coarse_reg_fpfhsac(const pcl::PointCloud<MullsPoint>::Ptr &source_cloud,
											 const pcl::PointCloud<MullsPoint>::Ptr &target_cloud,
											 pcl::PointCloud<MullsPoint>::Ptr &traned_source,
											 Eigen::Matrix4d &transformationS2T, float search_radius) {
	fpfhPtr source_fpfh(new fpfh());
	fpfhPtr target_fpfh(new fpfh());

	compute_fpfh_feature(source_cloud, source_fpfh, search_radius);
	compute_fpfh_feature(target_cloud, target_fpfh, search_radius);

	pcl::SampleConsensusInitialAlignment<MullsPoint, MullsPoint, pcl::FPFHSignature33> sac_ia;
	sac_ia.setInputSource(source_cloud);
	sac_ia.setSourceFeatures(source_fpfh);
	sac_ia.setInputTarget(target_cloud);
	sac_ia.setTargetFeatures(target_fpfh);
	//sac_ia.setNumberOfSamples(20);
	sac_ia.setCorrespondenceRandomness(15);
	sac_ia.align(*traned_source);
	transformationS2T = sac_ia.getFinalTransformation().template cast<double>();
	double fitness_score = sac_ia.getFitnessScore();
	return fitness_score;
}

bool MullsCalculate::find_feature_correspondence_ncc(const pcl::PointCloud<MullsPoint>::Ptr &target_kpts, const pcl::PointCloud<MullsPoint>::Ptr &source_kpts,
														pcl::PointCloud<MullsPoint>::Ptr &target_corrs, pcl::PointCloud<MullsPoint>::Ptr &source_corrs,
														bool fixed_num_corr, int corr_num, bool reciprocal_on) {
	// to enable reciprocal correspondence, you need to disable fixed_num_corr. 
	// once fixed_num_cor is enabled, reciprocal correspondence would be automatically disabled
	int target_kpts_num = target_kpts->points.size();
	int source_kpts_num = source_kpts->points.size();
	float dist_margin_thre = 0.0;

	LOG(INFO) << "MULLS : [" << target_kpts_num << "] key points in target point cloud and [" << source_kpts_num << "] key points in source point cloud.";

	if (target_kpts_num < 10 || source_kpts_num < 10) {
		LOG(WARNING) << "MULLS Too few key points\n";
		return false;
	}

	//first get descriptor
	std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> target_kpts_descriptors;
	float intensity_min = FLT_MAX;
	float intensity_max = 0;
	for (int i = 0; i < target_kpts_num; i++){
		float cur_i = target_kpts->points[i].intensity;
		intensity_min = std::min(intensity_min, cur_i);
		intensity_max = std::max(intensity_max, cur_i);
	}

	for (int i = 0; i < target_kpts_num; i++) {
		Eigen::VectorXf temp_descriptor(11);
		int temp_descriptor_close = (int)target_kpts->points[i].normal[0];
		int temp_descriptor_far = (int)target_kpts->points[i].normal[1];
		// neighborhood category with its distance to the query point
		temp_descriptor(0) = temp_descriptor_close / 1000000;
		temp_descriptor(1) = (temp_descriptor_close % 1000000) / 10000;
		temp_descriptor(2) = (temp_descriptor_close % 10000) / 100;
		temp_descriptor(3) = temp_descriptor_close % 100;
		temp_descriptor(4) = temp_descriptor_far / 1000000;
		temp_descriptor(5) = (temp_descriptor_far % 1000000) / 10000;
		temp_descriptor(6) = (temp_descriptor_far % 10000) / 100;
		temp_descriptor(7) = temp_descriptor_far % 100;
		// other properties
		float cur_i = target_kpts->points[i].intensity;
		temp_descriptor(8) = (cur_i - intensity_min) / (intensity_max - intensity_min) * 255.0; //[0 - 255] //normalized intensity 
		temp_descriptor(9) = target_kpts->points[i].data_c[3] * 100;							//[0 - 100] //curvature
		temp_descriptor(10) = target_kpts->points[i].data[3] * 30;								//[0 - 100] //height above ground
		//LOG(INFO) << temp_descriptor[1] << "," << temp_descriptor[2] << "," << temp_descriptor[3] << "," << temp_descriptor[4] << "," << temp_descriptor[5] << "," << temp_descriptor[6] << "," << temp_descriptor[7] << "," << temp_descriptor[8] << "," << temp_descriptor[9] << "," << temp_descriptor[10] << "," << temp_descriptor[11];
		target_kpts_descriptors.push_back(temp_descriptor);
	}

	std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> source_kpts_descriptors;
	for (int i = 0; i < source_kpts_num; i++) {
		Eigen::VectorXf temp_descriptor(11);
		int temp_descriptor_close = (int)source_kpts->points[i].normal[0];
		int temp_descriptor_far = (int)source_kpts->points[i].normal[1];
		// neighborhood category with its distance to the query point
		temp_descriptor(0) = temp_descriptor_close / 1000000;
		temp_descriptor(1) = (temp_descriptor_close % 1000000) / 10000;
		temp_descriptor(2) = (temp_descriptor_close % 10000) / 100;
		temp_descriptor(3) = temp_descriptor_close % 100;
		temp_descriptor(4) = temp_descriptor_far / 1000000;
		temp_descriptor(5) = (temp_descriptor_far % 1000000) / 10000;
		temp_descriptor(6) = (temp_descriptor_far % 10000) / 100;
		temp_descriptor(7) = temp_descriptor_far % 100;
		// other properties
		float cur_i = source_kpts->points[i].intensity;
		temp_descriptor(8) = (cur_i - intensity_min) / (intensity_max - intensity_min) * 255.0; //[0 - 255] //normalized intensity 
		temp_descriptor(9) = source_kpts->points[i].data_c[3] * 100; //[0 - 100] //curvature
		temp_descriptor(10) = source_kpts->points[i].data[3] * 30;   //[0 - 100] //height above ground
		//LOG(INFO) << temp_descriptor[1] << "," << temp_descriptor[2] << "," << temp_descriptor[3] << "," << temp_descriptor[4] << "," << temp_descriptor[5] << "," << temp_descriptor[6] << "," << temp_descriptor[7] << "," << temp_descriptor[8] << "," << temp_descriptor[9] << "," << temp_descriptor[10] << "," << temp_descriptor[11];
		source_kpts_descriptors.push_back(temp_descriptor);
	}

	std::vector<std::vector<float>> dist_table(target_kpts_num);
	for (int i = 0; i < target_kpts_num; i++)
		dist_table[i].resize(source_kpts_num);

	std::vector<std::pair<int, float>> dist_array;

	omp_set_num_threads(std::min(6, omp_get_max_threads()));
#pragma omp parallel for  //Multi-thread
	for (int i = 0; i < target_kpts_num; i++)
	{
		for (int j = 0; j < source_kpts_num; j++)
		{
			//Method 1. directly use L1 distance (use the features from 0 to 11)
			for (int k = 0; k < 11; k++)
				dist_table[i][j] += std::abs(target_kpts_descriptors[i](k) - source_kpts_descriptors[j](k));

			//Method 2. use cosine similarity instead
			//dist_table[i][j] =
			//target_kpts_descriptors[i].norm() * source_kpts_descriptors[j].norm() / target_kpts_descriptors[i].dot(source_kpts_descriptors[j]);

			//Method 3. use K-L divergence instead (use only the histogram (distribution)
			//for (int k = 0; k < 8; k++)
			//	dist_table[i][j] += 1.0 * target_kpts_descriptors[i](k) * std::log((1.0 * target_kpts_descriptors[i](k) + 0.001) / (1.0 * source_kpts_descriptors[j](k) + 0.001));
		}
	}
	if (!fixed_num_corr) {
		//find correspondence
		for (int i = 0; i < target_kpts_num; i++) {
			int min_dist_col_index = 0;
			float min_dist_row = FLT_MAX;
			for (int j = 0; j < source_kpts_num; j++) {
				if (dist_table[i][j] < min_dist_row) {
					min_dist_row = dist_table[i][j];
					min_dist_col_index = j;
				}
			}
			bool refined_corr = true;
			//reciprocal nearest neighbor correspondnece
			if (reciprocal_on) {
				for (int j = 0; j < target_kpts_num; j++) {
					if (min_dist_row > dist_table[j][min_dist_col_index] + dist_margin_thre) {
						refined_corr = false;
						break;
					}
				}
			}
			if (refined_corr) {
				target_corrs->points.push_back(target_kpts->points[i]);
				source_corrs->points.push_back(source_kpts->points[min_dist_col_index]);
			}
		}
	} else {
		//fixed num correspondence
		for (int i = 0; i < target_kpts_num; i++) {
			for (int j = 0; j < source_kpts_num; j++) {
				std::pair<int, float> temp_pair;
				temp_pair.first = i * source_kpts_num + j;
				temp_pair.second = dist_table[i][j];
				dist_array.push_back(temp_pair);
			}
		}
		std::sort(dist_array.begin(), dist_array.end(), [](const std::pair<int, float> &a, const std::pair<int, float> &b) { return a.second < b.second; });
		corr_num = std::min(corr_num, static_cast<int>(dist_array.size())); //take the k shortest distance

		std::vector<int> count_target_kpt(target_kpts_num, 0);
		std::vector<int> count_source_kpt(source_kpts_num, 0);

		int max_corr_num = 6;

		for (int k = 0; k < corr_num; k++) {
			int index = dist_array[k].first;
			int i = index / source_kpts_num;
			int j = index % source_kpts_num;

			if (count_target_kpt[i] > max_corr_num || count_source_kpt[j] > max_corr_num) //we only keep the first max_corr_num candidate correspondence of a single point in either source or target point cloud
				continue;

			count_target_kpt[i]++;
			count_source_kpt[j]++;

			target_corrs->points.push_back(target_kpts->points[i]);
			source_corrs->points.push_back(source_kpts->points[j]);
		}
	}

	//free memory
	std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>>().swap(target_kpts_descriptors);
	std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>>().swap(source_kpts_descriptors);
	std::vector<std::vector<float>>().swap(dist_table);
	std::vector<std::pair<int, float>>().swap(dist_array);
	return true;
}

int MullsCalculate::coarse_reg_ransac(const pcl::PointCloud<MullsPoint>::Ptr &target_pts,
						const pcl::PointCloud<MullsPoint>::Ptr &source_pts,
						Eigen::Matrix4d &tran_mat, float noise_bound, 
						int min_inlier_num, int max_iter_num) {		
	int N = target_pts->points.size();

	pcl::registration::CorrespondenceRejectorSampleConsensus<MullsPoint> ransac_rej;
	ransac_rej.setInputSource(source_pts);
	ransac_rej.setInputTarget(target_pts);
	ransac_rej.setInlierThreshold(noise_bound);
	ransac_rej.setMaximumIterations(max_iter_num);
	ransac_rej.setRefineModel(true);//false
	
	boost::shared_ptr<pcl::Correspondences> init_corres(new pcl::Correspondences);
	for (int i=0; i< N; i++) {
		pcl::Correspondence cur_corr;
		cur_corr.index_query=i;
		cur_corr.index_match=i;
		init_corres->push_back(cur_corr);
	}

	boost::shared_ptr<pcl::Correspondences> final_corres(new pcl::Correspondences);

	ransac_rej.setInputCorrespondences(init_corres);
	ransac_rej.getCorrespondences(*final_corres);

	if(final_corres->size() >= size_t(min_inlier_num)) {
		Eigen::Matrix4f best_tran =ransac_rej.getBestTransformation();
		tran_mat = best_tran.cast<double>();
		if (final_corres->size() >= 2 * size_t(min_inlier_num))
			return (1); //reliable
		else
			return (0); //need check
	} else {
		LOG(WARNING) << "Mulls Coarse RANSAC failed";
		return (-1);
	}
}

//coarse global registration using TEASER ++  (faster and more robust to outlier than RANSAC)
int MullsCalculate::coarse_reg_teaser(const pcl::PointCloud<MullsPoint>::Ptr &target_pts,
										 const pcl::PointCloud<MullsPoint>::Ptr &source_pts,
										 Eigen::Matrix4d &tran_mat, float noise_bound, int min_inlier_num) {
	int teaser_state = 0; //(failed: -1, successful[need check]: 0, successful[reliable]: 1)
	CHECK(target_pts->points.size() == source_pts->points.size());
	if (target_pts->points.size() <= 3) {
		LOG(WARNING) << "Teaser Correspondences Too Few";
		return (-1);
	}

	int N = target_pts->points.size();
	float min_inlier_ratio = 0.01;

	Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, N);
	Eigen::Matrix<double, 3, Eigen::Dynamic> tgt(3, N);

	for (int i = 0; i < N; ++i) {
		src.col(i) << source_pts->points[i].x, source_pts->points[i].y, source_pts->points[i].z;
		tgt.col(i) << target_pts->points[i].x, target_pts->points[i].y, target_pts->points[i].z;
	}

	// Run TEASER++ registration
	// Prepare solver parameters
	teaser::RobustRegistrationSolver::Params params;
	params.noise_bound = noise_bound;
	params.cbar2 = 1.0;
	params.estimate_scaling = false;
	params.rotation_max_iterations = 100;
	params.rotation_gnc_factor = 1.4;
	params.rotation_estimation_algorithm =
		teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
	params.use_max_clique = true;
	params.kcore_heuristic_threshold = 0.5;
	params.rotation_cost_threshold = 0.005; //1e-6

	// Solve with TEASER++
	teaser::RobustRegistrationSolver solver(params);
	std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
	solver.solve(src, tgt);
	std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
	std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);

	auto solution = solver.getSolution();
	std::vector<int> inliers;
	inliers = solver.getRotationInliers();
	if (solution.valid && inliers.size() >= size_t(min_inlier_num)) {
		tran_mat.setIdentity();
		tran_mat.block<3, 3>(0, 0) = solution.rotation;
		tran_mat.block<3, 1>(0, 3) = solution.translation;

		if (inliers.size() >= 2 * size_t(min_inlier_num))
			return (1); //reliable
		else
			return (0); //need check
	} else {
		LOG(WARNING) << "MULLS coarse Teaser failed";
		return (-1);
	}
	return (-1);
}

bool MullsCalculate::determine_source_target_cloud(const CloudBlockPtr &block_1, const CloudBlockPtr &block_2, 
                                                      Constraint &registration_cons) {
	if (block_1->down_feature_point_num > block_2->down_feature_point_num) {
		registration_cons.block1 = block_1;
		registration_cons.block2 = block_2;
	} else {
		registration_cons.block1 = block_2;
		registration_cons.block2 = block_1;
	}
	return true;
}

int MullsCalculate::mm_lls_icp(Constraint &registration_cons, int max_iter_num, float dis_thre_unit,
								  float converge_translation, float converge_rotation_d, float dis_thre_min,
								  float dis_thre_update_rate, std::string used_feature_type, std::string weight_strategy,
								  float z_xy_balanced_ratio, float pt2pt_residual_window, float pt2pl_residual_window,
								  float pt2li_residual_window, Eigen::Matrix4d initial_guess, bool apply_intersection_filter,
								  bool apply_motion_undistortion_while_registration, bool normal_shooting_on, float normal_bearing,
								  bool use_more_points, bool keep_less_source_points, float sigma_thre, float min_neccessary_corr_ratio, 
								  float max_bearable_rotation_d) {
	MullsFilter cfilter;
	//code that indicate the status of the registration
	//successful registration                ---   process_code= 1
	//too large tran or rot for one iter.    ---   process_code=-1
	//too few correspondences (ratio)        ---   process_code=-2
	//final standard deviation is too large  ---   process_code=-3
	int process_code = 0;

	//at least ${min_neccessary_corr_ratio} source points should have a match
	int min_total_corr_num = 40;
	int min_neccessary_corr_num = 20;

	float neccessary_corr_ratio = 1.0; //posterior unground points overlapping ratio

	Eigen::Matrix4d inv_init_guess_mat;
	Matrix6d cofactor_matrix;
	Matrix6d information_matrix;
	double sigma_square_post=1.0;
	Eigen::Matrix4d TempTran = Eigen::Matrix4d::Identity();
	Vector6d transform_x;

	cofactor_matrix.setIdentity();
	information_matrix.setIdentity();

	float dis_thre_ground = dis_thre_unit;
	float dis_thre_facade = dis_thre_unit;
	float dis_thre_roof = dis_thre_unit;
	float dis_thre_pillar = dis_thre_unit;
	float dis_thre_beam = dis_thre_unit;
	float dis_thre_vertex = dis_thre_unit;

	float max_bearable_translation = 2.0 * dis_thre_unit;
	float converge_rotation = converge_rotation_d / 180.0 * M_PI;
	float max_bearable_rotation = max_bearable_rotation_d / 180.0 * M_PI;

	//clone the point cloud
	pcl::PointCloud<MullsPoint>::Ptr pc_ground_sc(new pcl::PointCloud<MullsPoint>);
	pcl::PointCloud<MullsPoint>::Ptr pc_ground_tc(new pcl::PointCloud<MullsPoint>);
	pcl::PointCloud<MullsPoint>::Ptr pc_pillar_sc(new pcl::PointCloud<MullsPoint>);
	pcl::PointCloud<MullsPoint>::Ptr pc_pillar_tc(new pcl::PointCloud<MullsPoint>);
	pcl::PointCloud<MullsPoint>::Ptr pc_beam_sc(new pcl::PointCloud<MullsPoint>);
	pcl::PointCloud<MullsPoint>::Ptr pc_beam_tc(new pcl::PointCloud<MullsPoint>);
	pcl::PointCloud<MullsPoint>::Ptr pc_facade_sc(new pcl::PointCloud<MullsPoint>);
	pcl::PointCloud<MullsPoint>::Ptr pc_facade_tc(new pcl::PointCloud<MullsPoint>);
	pcl::PointCloud<MullsPoint>::Ptr pc_roof_sc(new pcl::PointCloud<MullsPoint>);
	pcl::PointCloud<MullsPoint>::Ptr pc_roof_tc(new pcl::PointCloud<MullsPoint>);
	pcl::PointCloud<MullsPoint>::Ptr pc_vertex_sc(new pcl::PointCloud<MullsPoint>);
	pcl::PointCloud<MullsPoint>::Ptr pc_vertex_tc(new pcl::PointCloud<MullsPoint>);

	registration_cons.block1->clone_feature(pc_ground_tc, pc_pillar_tc, pc_beam_tc, pc_facade_tc, pc_roof_tc, pc_vertex_tc, false);			   //target (would not change any more during the iteration)
	registration_cons.block2->clone_feature(pc_ground_sc, pc_pillar_sc, pc_beam_sc, pc_facade_sc, pc_roof_sc, pc_vertex_sc, !use_more_points); //source (point cloud used for a specific iteration)

	batch_transform_feature_points(pc_ground_sc, pc_pillar_sc, pc_beam_sc, pc_facade_sc, pc_roof_sc, pc_vertex_sc, initial_guess); //apply initial guess
	
	//Filter the point cloud laying far away from the overlapping region
	if (apply_intersection_filter && !apply_motion_undistortion_while_registration)
		intersection_filter(registration_cons, pc_ground_tc, pc_pillar_tc, pc_beam_tc, pc_facade_tc, pc_roof_tc, pc_vertex_tc,
							pc_ground_sc, pc_pillar_sc, pc_beam_sc, pc_facade_sc, pc_roof_sc, pc_vertex_sc);

	//Downsample source cloud if its point number is larger than target's
	if (keep_less_source_points && !apply_motion_undistortion_while_registration)
		keep_less_source_pts(pc_ground_tc, pc_pillar_tc, pc_beam_tc, pc_facade_tc, pc_roof_tc, pc_vertex_tc,
								pc_ground_sc, pc_pillar_sc, pc_beam_sc, pc_facade_sc, pc_roof_sc, pc_vertex_sc);

	int source_feature_points_count = 0;
	if (used_feature_type[1] == '1')
		source_feature_points_count += pc_pillar_sc->points.size();
	if (used_feature_type[2] == '1')
		source_feature_points_count += pc_facade_sc->points.size();
	if (used_feature_type[3] == '1')
		source_feature_points_count += pc_beam_sc->points.size();

	//Correspondence
	boost::shared_ptr<pcl::Correspondences> corrs_ground(new pcl::Correspondences),corrs_pillar(new pcl::Correspondences),corrs_beam(new pcl::Correspondences),corrs_facade(new pcl::Correspondences),corrs_roof(new pcl::Correspondences),corrs_vertex(new pcl::Correspondences);

	//build kd-tree in target point cloud
#pragma omp parallel sections
	{
#pragma omp section
		{
			if (used_feature_type[0] == '1' && pc_ground_tc->size() > 0)
				registration_cons.block1->tree_ground->setInputCloud(pc_ground_tc);
			if (used_feature_type[4] == '1' && pc_roof_tc->size() > 0)
				registration_cons.block1->tree_roof->setInputCloud(pc_roof_tc);
		}
#pragma omp section
		{
			if (used_feature_type[1] == '1' && pc_pillar_tc->size() > 0)
				registration_cons.block1->tree_pillar->setInputCloud(pc_pillar_tc);
			if (used_feature_type[3] == '1' && pc_beam_tc->size() > 0)
				registration_cons.block1->tree_beam->setInputCloud(pc_beam_tc);
		}
#pragma omp section
		{
			if (used_feature_type[2] == '1' && pc_facade_tc->size() > 0)
				registration_cons.block1->tree_facade->setInputCloud(pc_facade_tc);
		}
	}
	if (used_feature_type[5] == '1' && pc_vertex_tc->size() > 0)
		registration_cons.block1->tree_vertex->setInputCloud(pc_vertex_tc);

	//Iteration Loop
	for (int i = 0; i < max_iter_num; i++) {
		// Target (Dense): Cloud1,  Source (Sparse): Cloud2
		//apply motion undistortion
		inv_init_guess_mat = initial_guess.inverse();
		batch_transform_feature_points(pc_ground_sc, pc_pillar_sc, pc_beam_sc, pc_facade_sc, pc_roof_sc, pc_vertex_sc, TempTran);
		// First, find coorespondences (nearest neighbor [for line] or normal shooting [for plane])
		// We need to check the normal compatibility of the plane correspondences
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				if (used_feature_type[0] == '1' && pc_ground_sc->size() > 0)
					determine_corres(pc_ground_sc, pc_ground_tc, registration_cons.block1->tree_ground, dis_thre_ground, corrs_ground, normal_shooting_on, true, normal_bearing); //Check normal vector
			}			
			#pragma omp section
			{
				if (used_feature_type[1] == '1' && pc_pillar_sc->size() > 0)
					determine_corres(pc_pillar_sc, pc_pillar_tc, registration_cons.block1->tree_pillar, dis_thre_pillar, corrs_pillar, false, true, normal_bearing); //Check primary vector
			}
			#pragma omp section
			{
				if (used_feature_type[2] == '1' && pc_facade_sc->size() > 0)
					determine_corres(pc_facade_sc, pc_facade_tc, registration_cons.block1->tree_facade, dis_thre_facade, corrs_facade, normal_shooting_on, true, normal_bearing); //Check normal vector

				if (used_feature_type[3] == '1' && pc_beam_sc->size() > 0)
					determine_corres(pc_beam_sc, pc_beam_tc, registration_cons.block1->tree_beam, dis_thre_beam, corrs_beam, false, true, normal_bearing); //Check primary vector
			}
		}
		if (used_feature_type[4] == '1' && pc_roof_sc->size() > 0)
			determine_corres(pc_roof_sc, pc_roof_tc, registration_cons.block1->tree_roof, dis_thre_roof, corrs_roof, normal_shooting_on, true, normal_bearing); //Check normal vector
		if (used_feature_type[5] == '1' && pc_vertex_sc->size() > 0)
			determine_corres(pc_vertex_sc, pc_vertex_tc, registration_cons.block1->tree_vertex, dis_thre_vertex, corrs_vertex, false, false);
		int total_corr_num = (*corrs_ground).size() + (*corrs_pillar).size() + (*corrs_beam).size() + (*corrs_facade).size() + (*corrs_roof).size() + (*corrs_vertex).size();
		int neccessary_corr_num = (*corrs_pillar).size() + (*corrs_beam).size() + (*corrs_facade).size();
		neccessary_corr_ratio = 1.0 * neccessary_corr_num / source_feature_points_count;

		if (total_corr_num < min_total_corr_num || neccessary_corr_num < min_neccessary_corr_num || neccessary_corr_ratio < min_neccessary_corr_ratio)
		{
			process_code = -2;
			TempTran.setIdentity();
			LOG(WARNING) << "Mulls Fine Too few neccessary correspondences";
			break;
		}

		//update (decrease correspondence threshold)
		update_corr_dist_thre(dis_thre_ground, dis_thre_pillar, dis_thre_beam, dis_thre_facade, dis_thre_roof, dis_thre_vertex,
								dis_thre_update_rate, dis_thre_min);

		//Estimate Transformation
		multi_metrics_lls_tran_estimation(pc_ground_sc, pc_ground_tc, corrs_ground,
											pc_pillar_sc, pc_pillar_tc, corrs_pillar,
											pc_beam_sc, pc_beam_tc, corrs_beam,
											pc_facade_sc, pc_facade_tc, corrs_facade,
											pc_roof_sc, pc_roof_tc, corrs_roof,
											pc_vertex_sc, pc_vertex_tc, corrs_vertex,
											transform_x, cofactor_matrix,
											i, weight_strategy, z_xy_balanced_ratio,
											pt2pt_residual_window, pt2pl_residual_window, pt2li_residual_window);

		//About weight strategy:
		//0000: equal weight, // 1000: x,y,z balanced weight, //0100: residual weight, //0010: distance weight (adaptive), //0001: intensity weight	
		//....  //1111: all in
		
		//transform_x [6x1]: tx, ty, tz, roll, pitch, yaw --> Transformation matrix TempTran [4x4]
		construct_trans_a(transform_x(0), transform_x(1), transform_x(2), transform_x(3), transform_x(4), transform_x(5), TempTran);

		Eigen::Vector3d ts(TempTran(0, 3), TempTran(1, 3), TempTran(2, 3));
		Eigen::AngleAxisd rs(TempTran.block<3, 3>(0, 0));

		if (ts.norm() > max_bearable_translation || std::abs(rs.angle()) > max_bearable_rotation) {
			process_code = -1;
			TempTran.setIdentity();
			break;
		}

		//Judge converged or not
		if (i == max_iter_num - 1 || (i > 2 && ts.norm() < converge_translation && std::abs(rs.angle()) < converge_rotation)) {
			//Calculate converged residual and information matrix
			if (get_multi_metrics_lls_residual(pc_ground_sc, pc_ground_tc, corrs_ground,
												pc_pillar_sc, pc_pillar_tc, corrs_pillar,
												pc_beam_sc, pc_beam_tc, corrs_beam,
												pc_facade_sc, pc_facade_tc, corrs_facade,
												pc_roof_sc, pc_roof_tc, corrs_roof,
												pc_vertex_sc, pc_vertex_tc, corrs_vertex,
												transform_x, sigma_square_post, sigma_thre)) {
				process_code = 1;
			} else {
				process_code = -3;
			}

			//Calculate the information matrix
			//Background Knowledge:
			//For the adjustment problem : v=Ax-b
			//A is the design matrix, b is the observation matrix, x is the vector for estimation, P is the original weight matrix
			//Note that the information matrix is the inverse of the variance-covariance matrix ( Dxx ^ -1 ) of the estimated value (x,y,z,roll,pitch,yaw)
			//Dxx = Qxx * (sigma_post)^2 = ATPA * VTPV/(n-t) and Qxx = ATPA	
			//sigma_post^2 = (vTPv)/(n-t) = VTPV/(n-t)
			//v is the residual vector, n is the number of the observation equations and t is the number of the neccessary observation
			//information_matrix = (Dxx) ^(-1) =  (Qxx * (sigma_post)^2)^(-1) = ATPA / (sigma_post)^2  = (n-t)/(VTPV)*(ATPA)
			//because cofactor_matrix =  (ATPA)^(-1), so we get
			information_matrix = (1.0 / sigma_square_post) * cofactor_matrix.inverse();
			break; 
		}
		//Update the source pointcloud
		//batch_transform_feature_points(pc_ground_sc, pc_pillar_sc, pc_beam_sc, pc_facade_sc, pc_roof_sc, pc_vertex_sc, TempTran);

		//Update the transformation matrix till-now
		initial_guess = TempTran * initial_guess;
	}

	initial_guess = TempTran * initial_guess; //Update the last iteration's transformation

	registration_cons.Trans1_2 = initial_guess;

	//Multiple evalualtion metrics
	registration_cons.information_matrix = information_matrix; //Final information matrix
	registration_cons.sigma = std::sqrt(sigma_square_post);	//Final unit weight standard deviation
	registration_cons.confidence = neccessary_corr_ratio;	  //posterior unground points overlapping ratio
	//free mannually
	corrs_ground.reset(new pcl::Correspondences);
	corrs_pillar.reset(new pcl::Correspondences);
	corrs_beam.reset(new pcl::Correspondences);
	corrs_facade.reset(new pcl::Correspondences);
	corrs_roof.reset(new pcl::Correspondences);
	corrs_vertex.reset(new pcl::Correspondences);

	return process_code;
}

void MullsCalculate::batch_transform_feature_points(pcl::PointCloud<MullsPoint>::Ptr pc_ground, pcl::PointCloud<MullsPoint>::Ptr pc_pillar,
													   pcl::PointCloud<MullsPoint>::Ptr pc_beam, pcl::PointCloud<MullsPoint>::Ptr pc_facade,
													   pcl::PointCloud<MullsPoint>::Ptr pc_roof, pcl::PointCloud<MullsPoint>::Ptr pc_vertex,
													   Eigen::Matrix4d &Tran) {
	pcl::transformPointCloudWithNormals(*pc_ground, *pc_ground, Tran);
	pcl::transformPointCloudWithNormals(*pc_pillar, *pc_pillar, Tran);
	pcl::transformPointCloudWithNormals(*pc_beam, *pc_beam, Tran);
	pcl::transformPointCloudWithNormals(*pc_facade, *pc_facade, Tran);
	pcl::transformPointCloudWithNormals(*pc_roof, *pc_roof, Tran);
	pcl::transformPointCloudWithNormals(*pc_vertex, *pc_vertex, Tran);
}

//Time complexity of kdtree (in this case, the target point cloud [n points] is used for construct the tree while each point in source point cloud acts as a query point)
//build tree: O(nlogn) ---> so it's better to build the tree only once
//searching 1-nearest neighbor: O(logn) in average ---> so we can bear a larger number of target points
bool MullsCalculate::determine_corres(pcl::PointCloud<MullsPoint>::Ptr &Source_Cloud, pcl::PointCloud<MullsPoint>::Ptr &Target_Cloud,
										 const pcl::search::KdTree<MullsPoint>::Ptr &target_kdtree, float dis_thre,
										 boost::shared_ptr<pcl::Correspondences> &Corr_f, bool normal_shooting_on, bool normal_check,
										 float angle_thre_degree, bool duplicate_check, int K_filter_distant_point) {
	int K_min = 3;
	float filter_dis_times = 2.5;
	int normal_shooting_candidate_count = 10;

	pcl::registration::CorrespondenceEstimation<MullsPoint, MullsPoint> corr_est; //for nearest neighbor searching

	//CorrespondenceEstimationNormalShooting computes correspondences as points in the target cloud which have minimum distance to normals computed on the input cloud
	pcl::registration::CorrespondenceEstimationNormalShooting<MullsPoint, MullsPoint, MullsPoint> corr_est_ns; //for normal shooting searching

	pcl::registration::CorrespondenceRejectorDistance corr_rej_dist;

	boost::shared_ptr<pcl::Correspondences> Corr(new pcl::Correspondences);

	pcl::PointCloud<MullsPoint>::Ptr Source_Cloud_f(new pcl::PointCloud<MullsPoint>);
	pcl::PointCloud<MullsPoint>::Ptr Target_Cloud_f(new pcl::PointCloud<MullsPoint>); //target point cloud would never change

	if (Source_Cloud->points.size() >= size_t(K_min) &&
		Target_Cloud->points.size() >= size_t(K_min)) {
		// Normal Shooting
		if (normal_shooting_on) {
			corr_est_ns.setInputSource(Source_Cloud);
			corr_est_ns.setInputTarget(Target_Cloud);
			corr_est_ns.setSourceNormals(Source_Cloud);
			corr_est_ns.setSearchMethodTarget(target_kdtree, true);					  //saving the time of rebuilding kd-tree
			corr_est_ns.setKSearch(normal_shooting_candidate_count);				  // Among the K nearest neighbours find the one with minimum perpendicular distance to the normal
			corr_est_ns.determineCorrespondences(*Corr, filter_dis_times * dis_thre); //base on KDtreeNSearch
																						//corr_est_ns.determineReciprocalCorrespondences(*Corr);
		} else {
			//Nearest Neighbor
			corr_est.setInputCloud(Source_Cloud);
			corr_est.setInputTarget(Target_Cloud);
			corr_est.setSearchMethodTarget(target_kdtree, true);				   //saving the time of rebuilding kd-tree
			corr_est.determineCorrespondences(*Corr, filter_dis_times * dis_thre); //base on KDtreeNSearch
																					//corr_est.determineReciprocalCorrespondences(*Corr);
		}

		if (Source_Cloud->points.size() >= size_t(K_filter_distant_point)) {
			int count = 0;

			//duplicate check -> just keep one source point corresponding to one target point
			std::vector<unsigned int> duplicate_check_table(Target_Cloud->points.size(), 0);

			for (auto iter = Corr->begin(); iter != Corr->end();) {
				int s_index, t_index;
				s_index = (*iter).index_query;
				t_index = (*iter).index_match;

				if (t_index != -1) {
					if (duplicate_check && duplicate_check_table[t_index] > 0)
						iter = Corr->erase(iter);
					else {
						duplicate_check_table[t_index]++;

						Source_Cloud_f->points.push_back(Source_Cloud->points[s_index]);
						(*iter).index_query = count;
						count++;
						iter++;
					}
				}
				else
					iter++;
			}
			Corr->resize(count);

			Source_Cloud_f->points.swap(Source_Cloud->points);
			std::vector<unsigned int>().swap(duplicate_check_table);
		}
		corr_rej_dist.setInputCorrespondences(Corr);
		corr_rej_dist.setMaximumDistance(dis_thre);
		corr_rej_dist.getCorrespondences(*Corr_f);

		//only for planar points
		if (normal_check) {
			int count = 0;
			//Normal direction consistency check
			for (auto iter = Corr_f->begin(); iter != Corr_f->end();) {
				int s_index, t_index;
				s_index = (*iter).index_query;
				t_index = (*iter).index_match;

				if (t_index != -1) {
					Eigen::Vector3d n1;
					Eigen::Vector3d n2;
					n1 << Source_Cloud->points[s_index].normal[0], Source_Cloud->points[s_index].normal[1], Source_Cloud->points[s_index].normal[2];
					n2 << Target_Cloud->points[t_index].normal[0], Target_Cloud->points[t_index].normal[1], Target_Cloud->points[t_index].normal[2];

					float cos_intersection_angle = std::abs(n1.dot(n2)); // n1.norm()=n2.norm()=1

					if (cos_intersection_angle < cos(angle_thre_degree / 180.0 * M_PI)) {
						count++;
						iter = Corr_f->erase(iter);
					}
					else
						iter++;
				}
				else
					iter++;
			}
		}
	}
	else
		return false;
	return true;
}


void MullsCalculate::update_corr_dist_thre(float &dis_thre_ground, float &dis_thre_pillar, float &dis_thre_beam,
											  float &dis_thre_facade, float &dis_thre_roof, float &dis_thre_vertex,
											  float dis_thre_update_rate, float dis_thre_min) {
	dis_thre_ground = std::max(1.0f * dis_thre_ground / dis_thre_update_rate, dis_thre_min);
	dis_thre_facade = std::max(1.0f * dis_thre_facade / dis_thre_update_rate, dis_thre_min);
	dis_thre_roof = std::max(1.0f * dis_thre_roof / dis_thre_update_rate, dis_thre_min);
	dis_thre_pillar = std::max(1.0f * dis_thre_pillar / dis_thre_update_rate, dis_thre_min);
	dis_thre_beam = std::max(1.0f * dis_thre_beam / dis_thre_update_rate, dis_thre_min);
	dis_thre_vertex = std::max(1.0f * dis_thre_vertex / dis_thre_update_rate, dis_thre_min);
}

bool MullsCalculate::multi_metrics_lls_tran_estimation(const pcl::PointCloud<MullsPoint>::Ptr &Source_Ground, const pcl::PointCloud<MullsPoint>::Ptr &Target_Ground, boost::shared_ptr<pcl::Correspondences> &Corr_Ground,
														  const pcl::PointCloud<MullsPoint>::Ptr &Source_Pillar, const pcl::PointCloud<MullsPoint>::Ptr &Target_Pillar, boost::shared_ptr<pcl::Correspondences> &Corr_Pillar,
														  const pcl::PointCloud<MullsPoint>::Ptr &Source_Beam, const pcl::PointCloud<MullsPoint>::Ptr &Target_Beam, boost::shared_ptr<pcl::Correspondences> &Corr_Beam,
														  const pcl::PointCloud<MullsPoint>::Ptr &Source_Facade, const pcl::PointCloud<MullsPoint>::Ptr &Target_Facade, boost::shared_ptr<pcl::Correspondences> &Corr_Facade,
														  const pcl::PointCloud<MullsPoint>::Ptr &Source_Roof, const pcl::PointCloud<MullsPoint>::Ptr &Target_Roof, boost::shared_ptr<pcl::Correspondences> &Corr_Roof,
														  const pcl::PointCloud<MullsPoint>::Ptr &Source_Vertex, const pcl::PointCloud<MullsPoint>::Ptr &Target_Vertex, boost::shared_ptr<pcl::Correspondences> &Corr_Vertex,
														  Vector6d &unknown_x, Matrix6d &cofactor_matrix, int iter_num, std::string weight_strategy, float z_xy_balance_ratio, float pt2pt_residual_window,
														  float pt2pl_residual_window, float pt2li_residual_window) {
	Matrix6d ATPA;
	Vector6d ATPb;
	ATPA.setZero();
	ATPb.setZero();

	//Deal with weight (contribution of each correspondence in the transformation estimation)
	float w_ground = 1.0, w_facade = 1.0, w_roof = 1.0, w_pillar = 1.0, w_beam = 1.0, w_vertex = 1.0; //initialization

	int m1 = (*Corr_Ground).size() + (*Corr_Roof).size();
	int m2 = (*Corr_Facade).size();
	int m3 = (*Corr_Pillar).size();
	int m4 = (*Corr_Beam).size();
	int m5 = (*Corr_Vertex).size();

	//x,y,z directional balanced weighting (guarantee the observability of the scene)
	if (weight_strategy[0] == '1') {
		w_ground = std::max(0.01, z_xy_balance_ratio * (m2 + 2 * m3 - m4) / (0.0001 + 2.0 * m1)); // x <-> y <-> z
		w_roof = w_ground;
		w_facade = 1.0;
		w_pillar = 1.0;
		w_beam = 1.0;
		w_vertex = 1.0;
	}

	bool dist_weight = false;
	bool residual_weight = false;
	bool intensity_weight = false;
	int iter_thre = 2; //the residual based weighting would only be applied after this number of iteration
	if (weight_strategy[1] == '1' && iter_num > iter_thre) //weight according to residual 
		residual_weight = true;
	if (weight_strategy[2] == '1') //weight according to distance
		dist_weight = true;
	if (weight_strategy[3] == '1') //weight according to intensity
		intensity_weight = true;

	//point to plane
	pt2pl_lls_summation(Source_Ground, Target_Ground, Corr_Ground, ATPA, ATPb, iter_num, w_ground, dist_weight, residual_weight, intensity_weight, pt2pl_residual_window);
	pt2pl_lls_summation(Source_Facade, Target_Facade, Corr_Facade, ATPA, ATPb, iter_num, w_facade, dist_weight, residual_weight, intensity_weight, pt2pl_residual_window);
	pt2pl_lls_summation(Source_Roof, Target_Roof, Corr_Roof, ATPA, ATPb, iter_num, w_roof, dist_weight, residual_weight, intensity_weight, pt2pl_residual_window);
	//point to line
	pt2li_lls_pri_direction_summation(Source_Pillar, Target_Pillar, Corr_Pillar, ATPA, ATPb, iter_num, w_pillar, dist_weight, residual_weight, intensity_weight, pt2li_residual_window);
	pt2li_lls_pri_direction_summation(Source_Beam, Target_Beam, Corr_Beam, ATPA, ATPb, iter_num, w_beam, dist_weight, residual_weight, intensity_weight, pt2li_residual_window);
	//point to point
	pt2pt_lls_summation(Source_Vertex, Target_Vertex, Corr_Vertex, ATPA, ATPb, iter_num, w_vertex, dist_weight, residual_weight, intensity_weight, pt2pt_residual_window);

	//ATPA is a symmetric matrix
	ATPA.coeffRef(6) = ATPA.coeffRef(1);
	ATPA.coeffRef(12) = ATPA.coeffRef(2);
	ATPA.coeffRef(13) = ATPA.coeffRef(8);
	ATPA.coeffRef(18) = ATPA.coeffRef(3);
	ATPA.coeffRef(19) = ATPA.coeffRef(9);
	ATPA.coeffRef(20) = ATPA.coeffRef(15);
	ATPA.coeffRef(24) = ATPA.coeffRef(4);
	ATPA.coeffRef(25) = ATPA.coeffRef(10);
	ATPA.coeffRef(26) = ATPA.coeffRef(16);
	ATPA.coeffRef(27) = ATPA.coeffRef(22);
	ATPA.coeffRef(30) = ATPA.coeffRef(5);
	ATPA.coeffRef(31) = ATPA.coeffRef(11);
	ATPA.coeffRef(32) = ATPA.coeffRef(17);
	ATPA.coeffRef(33) = ATPA.coeffRef(23);
	ATPA.coeffRef(34) = ATPA.coeffRef(29);

	//LOG(INFO) << "ATPA=" << std::endl << ATPA;
	//LOG(INFO) << "ATPb=" << std::endl << ATPb;

	// Solve A*x = b  x= (ATPA)^(-1)ATPb
	// x: tx ty tz alpha beta gamma (alpha beta gamma corresponding to roll, pitch and yaw)
	// the approximated rotation matrix is
	// |   1    -gamma   beta  |
	// | gamma     1    -alpha |
	// | -beta   alpha     1   |
	//reference: A Review of Point Cloud Registration Algorithms for Mobile Robotics, Appendix

	unknown_x = ATPA.inverse() * ATPb;

	Eigen::Vector3d euler_angle(unknown_x(3), unknown_x(4), unknown_x(5));
	Eigen::Matrix3d Jacobi;
	get_quat_euler_jacobi(euler_angle, Jacobi);

	//Qxx=(ATPA)^-1
	//information matrix = Dxx^(-1)=Qxx^(-1)/(sigma_post)^2=ATPA/(sigma_post)^2
	cofactor_matrix = ATPA.inverse();

	//convert to the cofactor matrix with regard to quaternion from euler angle
	cofactor_matrix.block<3, 3>(3, 3) = Jacobi * cofactor_matrix.block<3, 3>(3, 3) * Jacobi.transpose();
	cofactor_matrix.block<3, 3>(0, 3) = cofactor_matrix.block<3, 3>(0, 3) * Jacobi.transpose();
	cofactor_matrix.block<3, 3>(3, 0) = Jacobi * cofactor_matrix.block<3, 3>(3, 0);

	return true;
}


bool MullsCalculate::pt2pt_lls_summation(const pcl::PointCloud<MullsPoint>::Ptr &Source_Cloud, const pcl::PointCloud<MullsPoint>::Ptr &Target_Cloud,
											boost::shared_ptr<pcl::Correspondences> &Corr, Matrix6d &ATPA, Vector6d &ATPb, int iter_num,
											float weight, bool dist_weight_or_not , bool residual_weight_or_not, bool intensity_weight_or_not,
											float residual_window_size) {
	for (size_t i = 0u; i < (*Corr).size(); i++) {
		int s_index, t_index;
		s_index = (*Corr)[i].index_query;
		t_index = (*Corr)[i].index_match;

		if (t_index != -1) {
			float px = Source_Cloud->points[s_index].x;
			float py = Source_Cloud->points[s_index].y;
			float pz = Source_Cloud->points[s_index].z;
			float qx = Target_Cloud->points[t_index].x;
			float qy = Target_Cloud->points[t_index].y;
			float qz = Target_Cloud->points[t_index].z;

			float pi = Source_Cloud->points[s_index].intensity;
			float qi = Target_Cloud->points[t_index].intensity;

			float dx = px - qx;
			float dy = py - qy;
			float dz = pz - qz;

			float wx, wy, wz;
			wx = weight;

			float dist = std::sqrt(qx * qx + qy * qy + qz * qz);

			if (dist_weight_or_not)
				wx = wx * get_weight_by_dist_adaptive(dist, iter_num);
			//wx = wx * get_weight_by_dist(dist);

			if (residual_weight_or_not)
				wx = wx * get_weight_by_residual(std::sqrt(dx * dx + dy * dy + dz * dz), residual_window_size);
			if (intensity_weight_or_not)
				wx = wx * get_weight_by_intensity(pi + 0.0001, qi + 0.0001);

			wy = wx;
			wz = wx;
			
			// unknown x: [tx ty tz alpha beta gama] 

			//    0  1  2  3  4  5
			//    6  7  8  9 10 11
			//   12 13 14 15 16 17
			//   18 19 20 21 22 23
			//   24 25 26 27 28 29
			//   30 31 32 33 34 35

			ATPA.coeffRef(0) += wx;
			ATPA.coeffRef(1) += 0;
			ATPA.coeffRef(2) += 0;
			ATPA.coeffRef(3) += 0;
			ATPA.coeffRef(4) += wx * pz;
			ATPA.coeffRef(5) += (-wx * py);
			ATPA.coeffRef(7) += wy;
			ATPA.coeffRef(8) += 0;
			ATPA.coeffRef(9) += (-wy * pz);
			ATPA.coeffRef(10) += 0;
			ATPA.coeffRef(11) += wy * px;
			ATPA.coeffRef(14) += wz;
			ATPA.coeffRef(15) += wz * py;
			ATPA.coeffRef(16) += (-wz * px);
			ATPA.coeffRef(17) += 0;
			ATPA.coeffRef(21) += wy * pz * pz + wz * py * py;
			ATPA.coeffRef(22) += (-wz * px * py);
			ATPA.coeffRef(23) += (-wy * px * pz);
			ATPA.coeffRef(28) += wx * pz * pz + wz * px * px;
			ATPA.coeffRef(29) += (-wx * py * pz);
			ATPA.coeffRef(35) += wx * py * py + wy * px * px;

			ATPb.coeffRef(0) += (-wx * dx);
			ATPb.coeffRef(1) += (-wy * dy);
			ATPb.coeffRef(2) += (-wz * dz);
			ATPb.coeffRef(3) += wy * pz * dy - wz * py * dz;
			ATPb.coeffRef(4) += wz * px * dz - wx * pz * dx;
			ATPb.coeffRef(5) += wx * py * dx - wy * px * dy;
		}
	}
	return true;
}

bool MullsCalculate::pt2pl_lls_summation(const pcl::PointCloud<MullsPoint>::Ptr &Source_Cloud, const pcl::PointCloud<MullsPoint>::Ptr &Target_Cloud,
											boost::shared_ptr<pcl::Correspondences> &Corr, Matrix6d &ATPA, Vector6d &ATPb, int iter_num, float weight, 
											bool dist_weight_or_not, bool residual_weight_or_not, bool intensity_weight_or_not, float residual_window_size) {
	for (size_t i = 0u; i < (*Corr).size(); i++) {
		int s_index, t_index;
		s_index = (*Corr)[i].index_query;
		t_index = (*Corr)[i].index_match;

		if (t_index != -1) {
			float px = Source_Cloud->points[s_index].x;
			float py = Source_Cloud->points[s_index].y;
			float pz = Source_Cloud->points[s_index].z;
			float qx = Target_Cloud->points[t_index].x;
			float qy = Target_Cloud->points[t_index].y;
			float qz = Target_Cloud->points[t_index].z;
			float ntx = Target_Cloud->points[t_index].normal_x;
			float nty = Target_Cloud->points[t_index].normal_y;
			float ntz = Target_Cloud->points[t_index].normal_z;

			float pi = Source_Cloud->points[s_index].intensity;
			float qi = Target_Cloud->points[t_index].intensity;

			float w = weight;

			float a = ntz * py - nty * pz;
			float b = ntx * pz - ntz * px;
			float c = nty * px - ntx * py;

			float d = ntx * qx + nty * qy + ntz * qz - ntx * px - nty * py - ntz * pz;

			float dist = std::sqrt(qx * qx + qy * qy + qz * qz);

			if (dist_weight_or_not)
				w = w * get_weight_by_dist_adaptive(dist, iter_num);
			//w = w * get_weight_by_dist(dist);

			if (residual_weight_or_not)
				w = w * get_weight_by_residual(std::abs(d), residual_window_size);
			//w = w * get_weight_by_residual_general(std::abs(d), residual_window_size, 1.0);

			if (intensity_weight_or_not)
				w = w * get_weight_by_intensity(pi + 0.0001, qi + 0.0001);

			(*Corr)[i].weight = w;

			//    0  1  2  3  4  5
			//    6  7  8  9 10 11
			//   12 13 14 15 16 17
			//   18 19 20 21 22 23
			//   24 25 26 27 28 29
			//   30 31 32 33 34 35

			ATPA.coeffRef(0) += w * ntx * ntx;
			ATPA.coeffRef(1) += w * ntx * nty;
			ATPA.coeffRef(2) += w * ntx * ntz;
			ATPA.coeffRef(3) += w * a * ntx;
			ATPA.coeffRef(4) += w * b * ntx;
			ATPA.coeffRef(5) += w * c * ntx;
			ATPA.coeffRef(7) += w * nty * nty;
			ATPA.coeffRef(8) += w * nty * ntz;
			ATPA.coeffRef(9) += w * a * nty;
			ATPA.coeffRef(10) += w * b * nty;
			ATPA.coeffRef(11) += w * c * nty;
			ATPA.coeffRef(14) += w * ntz * ntz;
			ATPA.coeffRef(15) += w * a * ntz;
			ATPA.coeffRef(16) += w * b * ntz;
			ATPA.coeffRef(17) += w * c * ntz;
			ATPA.coeffRef(21) += w * a * a;
			ATPA.coeffRef(22) += w * a * b;
			ATPA.coeffRef(23) += w * a * c;
			ATPA.coeffRef(28) += w * b * b;
			ATPA.coeffRef(29) += w * b * c;
			ATPA.coeffRef(35) += w * c * c;

			ATPb.coeffRef(0) += w * d * ntx;
			ATPb.coeffRef(1) += w * d * nty;
			ATPb.coeffRef(2) += w * d * ntz;
			ATPb.coeffRef(3) += w * d * a;
			ATPb.coeffRef(4) += w * d * b;
			ATPb.coeffRef(5) += w * d * c;
		}
	}
	return true;
}

bool MullsCalculate::pt2li_lls_pri_direction_summation(const pcl::PointCloud<MullsPoint>::Ptr &Source_Cloud, const pcl::PointCloud<MullsPoint>::Ptr &Target_Cloud,
														  boost::shared_ptr<pcl::Correspondences> &Corr, Matrix6d &ATPA, Vector6d &ATPb, int iter_num,
														  float weight, bool dist_weight_or_not, bool residual_weight_or_not, bool intensity_weight_or_not,
														  float residual_window_size) {
	for (size_t i = 0u; i < (*Corr).size(); i++) {
		int s_index, t_index;
		s_index = (*Corr)[i].index_query;
		t_index = (*Corr)[i].index_match;

		if (t_index != -1) {
			float px = Source_Cloud->points[s_index].x;
			float py = Source_Cloud->points[s_index].y;
			float pz = Source_Cloud->points[s_index].z;
			float qx = Target_Cloud->points[t_index].x;
			float qy = Target_Cloud->points[t_index].y;
			float qz = Target_Cloud->points[t_index].z;

			//primary direction (in this case, we save the primary direction of linear feature points in the normal vector)
			float vx = Target_Cloud->points[t_index].normal_x;
			float vy = Target_Cloud->points[t_index].normal_y;
			float vz = Target_Cloud->points[t_index].normal_z;

			//LOG(INFO) << nx << "," << ny<< "," <<nz;

			float pi = Source_Cloud->points[s_index].intensity;
			float qi = Target_Cloud->points[t_index].intensity;

			float dx = px - qx;
			float dy = py - qy;
			float dz = pz - qz;

			Eigen::Matrix<double, 3, 6> Amat;
			Eigen::Matrix<double, 3, 1> bvec;
			Eigen::Matrix<double, 3, 1> evec;
			Eigen::Matrix<double, 3, 3> Imat;
			Eigen::Matrix<double, 3, 3> Wmat;
			Imat.setIdentity();
			Wmat.setIdentity();

			Amat(0, 0) = 0;
			Amat(0, 1) = -vz;
			Amat(0, 2) = vy;
			Amat(0, 3) = vy * py + vz * pz;
			Amat(0, 4) = -vy * px;
			Amat(0, 5) = -vz * px;
			Amat(1, 0) = vz;
			Amat(1, 1) = 0;
			Amat(1, 2) = -vx;
			Amat(1, 3) = -vx * py;
			Amat(1, 4) = vz * pz + vx * px;
			Amat(1, 5) = -vz * py;
			Amat(2, 0) = -vy;
			Amat(2, 1) = vx;
			Amat(2, 2) = 0;
			Amat(2, 3) = -vx * pz;
			Amat(2, 4) = -vy * pz;
			Amat(2, 5) = vx * px + vy * py;

			bvec(0, 0) = -vy * dz + vz * dy;
			bvec(1, 0) = -vz * dx + vx * dz;
			bvec(2, 0) = -vx * dy + vy * dx;

			//evec = (Amat * (Amat.transpose() * Amat).inverse() * Amat.transpose() - Imat) * bvec; //posterior residual
			//we'd like to directly use the prior residual
			float ex = std::abs(bvec(0, 0));
			float ey = std::abs(bvec(1, 0));
			float ez = std::abs(bvec(2, 0));
			float ed = std::sqrt(ex * ex + ey * ey + ez * ez);

			float wx, wy, wz, w;
			wx = weight;

			float dist = std::sqrt(qx * qx + qy * qy + qz * qz);

			if (dist_weight_or_not)
				//wx *= get_weight_by_dist(dist);
				wx *= get_weight_by_dist_adaptive(dist, iter_num);

			if (intensity_weight_or_not)
				wx *= get_weight_by_intensity(pi + 0.0001, qi + 0.0001);

			if (residual_weight_or_not) {
				wx = wx * get_weight_by_residual(ed, residual_window_size); //original huber
																			// wx = wx * get_weight_by_residual_general(ed, residual_window_size, 1.0);
			}
			wy = wx;
			wz = wx;
			(*Corr)[i].weight = wx;

			Wmat(0, 0) = std::sqrt(wx);
			Wmat(1, 1) = std::sqrt(wy);
			Wmat(2, 2) = std::sqrt(wz);

			Amat = Wmat * Amat;
			bvec = Wmat * bvec;

			for (int j = 0; j < 6; j++) {
				for (int k = j; k < 6; k++)
					ATPA(j, k) += ((Amat.block<3, 1>(0, j)).transpose() * (Amat.block<3, 1>(0, k)));
			}
			for (int j = 0; j < 6; j++)
				ATPb.coeffRef(j) += ((Amat.block<3, 1>(0, j)).transpose() * (bvec));
		}
	}
	return true;
}


bool MullsCalculate::ground_3dof_lls_tran_estimation(const pcl::PointCloud<MullsPoint>::Ptr &Source_Ground,
														const pcl::PointCloud<MullsPoint>::Ptr &Target_Ground,
														boost::shared_ptr<pcl::Correspondences> &Corr_Ground,
														Eigen::Vector3d &unknown_x, Eigen::Matrix3d &cofactor_matrix,
														int iter_num, std::string weight_strategy) {
	Eigen::Matrix3d ATPA;
	Eigen::Vector3d ATPb;
	ATPA.setZero();
	ATPb.setZero();

	bool dist_weight = false;
	bool residual_weight = false;
	bool intensity_weight = false;

	if (weight_strategy[1] == '1') //weight according to residual
		residual_weight = true;

	if (weight_strategy[2] == '1') //weight according to distance
		dist_weight = true;

	if (weight_strategy[3] == '1') //weight according to intensity
		intensity_weight = true;

	pt2pl_ground_3dof_lls_summation(Source_Ground, Target_Ground, Corr_Ground, ATPA, ATPb, iter_num, 1.0, dist_weight, residual_weight, intensity_weight);

	//ATPA is a symmetric matrix
	//    0  1  2
	//   [3] 4  5
	//   [6][7] 8
	ATPA.coeffRef(3) = ATPA.coeffRef(1);
	ATPA.coeffRef(6) = ATPA.coeffRef(2);
	ATPA.coeffRef(7) = ATPA.coeffRef(5);

	// Solve A*x = b  x= (ATPA)^(-1)ATPb
	// x: tx ty tz alpha beta gamma
	unknown_x = ATPA.inverse() * ATPb;

	return true;
}

bool MullsCalculate::pt2pl_ground_3dof_lls_summation(const pcl::PointCloud<MullsPoint>::Ptr &Source_Cloud, const pcl::PointCloud<MullsPoint>::Ptr &Target_Cloud,
														boost::shared_ptr<pcl::Correspondences> &Corr, Eigen::Matrix3d &ATPA, Eigen::Vector3d &ATPb, int iter_num,
														float weight, bool dist_weight_or_not, bool residual_weight_or_not , bool intensity_weight_or_not,
														float residual_window_size) {
	//unknown : roll (alpha), picth (beta) and tz
	for (size_t i = 0u; i < (*Corr).size(); i++)
	{
		int s_index, t_index;
		s_index = (*Corr)[i].index_query;
		t_index = (*Corr)[i].index_match;

		if (t_index != -1)
		{

			float px = Source_Cloud->points[s_index].x;
			float py = Source_Cloud->points[s_index].y;
			float pz = Source_Cloud->points[s_index].z;
			float qx = Target_Cloud->points[t_index].x;
			float qy = Target_Cloud->points[t_index].y;
			float qz = Target_Cloud->points[t_index].z;
			float ntx = Target_Cloud->points[t_index].normal_x;
			float nty = Target_Cloud->points[t_index].normal_y;
			float ntz = Target_Cloud->points[t_index].normal_z;

			float pi = Source_Cloud->points[s_index].intensity;
			float qi = Target_Cloud->points[t_index].intensity;

			float w = weight;

			float a = ntz * py - nty * pz;
			float b = ntx * pz - ntz * px;

			float d = ntx * qx + nty * qy + ntz * qz - ntx * px - nty * py - ntz * pz;

			float dist = std::sqrt(qx * qx + qy * qy + qz * qz);
			if (dist_weight_or_not)
				w = w * get_weight_by_dist_adaptive(dist, iter_num);
			//w = w * get_weight_by_dist(dist);

			if (residual_weight_or_not)
				w = w * get_weight_by_residual(std::abs(d), residual_window_size);

			if (intensity_weight_or_not)
				w = w * get_weight_by_intensity(pi + 0.0001, qi + 0.0001);

			//    0  1  2
			//    3  4  5
			//    6  7  8

			(*Corr)[i].weight = w;

			ATPA.coeffRef(0) += w * a * a;
			ATPA.coeffRef(1) += w * a * b;
			ATPA.coeffRef(2) += w * a * ntz;
			ATPA.coeffRef(4) += w * b * b;
			ATPA.coeffRef(5) += w * b * ntz;
			ATPA.coeffRef(8) += w * ntz * ntz;

			ATPb.coeffRef(0) += w * d * a;
			ATPb.coeffRef(1) += w * d * b;
			ATPb.coeffRef(2) += w * d * ntz;
		}
	}

	return true;
}

bool MullsCalculate::get_multi_metrics_lls_residual(const pcl::PointCloud<MullsPoint>::Ptr &Source_Ground, const pcl::PointCloud<MullsPoint>::Ptr &Target_Ground, boost::shared_ptr<pcl::Correspondences> &Corr_Ground,
													   const pcl::PointCloud<MullsPoint>::Ptr &Source_Pillar, const pcl::PointCloud<MullsPoint>::Ptr &Target_Pillar, boost::shared_ptr<pcl::Correspondences> &Corr_Pillar,
													   const pcl::PointCloud<MullsPoint>::Ptr &Source_Beam, const pcl::PointCloud<MullsPoint>::Ptr &Target_Beam, boost::shared_ptr<pcl::Correspondences> &Corr_Beam,
													   const pcl::PointCloud<MullsPoint>::Ptr &Source_Facade, const pcl::PointCloud<MullsPoint>::Ptr &Target_Facade, boost::shared_ptr<pcl::Correspondences> &Corr_Facade,
													   const pcl::PointCloud<MullsPoint>::Ptr &Source_Roof, const pcl::PointCloud<MullsPoint>::Ptr &Target_Roof, boost::shared_ptr<pcl::Correspondences> &Corr_Roof,
													   const pcl::PointCloud<MullsPoint>::Ptr &Source_Vertex, const pcl::PointCloud<MullsPoint>::Ptr &Target_Vertex, boost::shared_ptr<pcl::Correspondences> &Corr_Vertex,
													   const Vector6d &transform_x, double &sigma_square_post, double sigma_thre) {
	double VTPV = 0;
	int obeservation_count = 0;

	pt2pl_lls_residual(Source_Ground, Target_Ground, Corr_Ground, transform_x, VTPV, obeservation_count);
	pt2pl_lls_residual(Source_Facade, Target_Facade, Corr_Facade, transform_x, VTPV, obeservation_count);
	pt2pl_lls_residual(Source_Roof, Target_Roof, Corr_Roof, transform_x, VTPV, obeservation_count);
	pt2li_lls_residual(Source_Pillar, Target_Pillar, Corr_Pillar, transform_x, VTPV, obeservation_count);
	pt2li_lls_residual(Source_Beam, Target_Beam, Corr_Beam, transform_x, VTPV, obeservation_count);
	pt2pt_lls_residual(Source_Vertex, Target_Vertex, Corr_Vertex, transform_x, VTPV, obeservation_count);

	sigma_square_post = VTPV / (obeservation_count - 6); //   VTPV/(n-t) , t is the neccessary observation number (dof), here, t=6

	if (sqrt(sigma_square_post) < sigma_thre)
		return true;
	else
		return false;
}

bool MullsCalculate::pt2pt_lls_residual(const pcl::PointCloud<MullsPoint>::Ptr &Source_Cloud, const pcl::PointCloud<MullsPoint>::Ptr &Target_Cloud,
						                   boost::shared_ptr<pcl::Correspondences> &Corr, const Vector6d &transform_x, double &VTPV, int &observation_count) {
	//point-to-plane distance metrics
	//3 observation equation for 1 pair of correspondence
	for (size_t i = 0u; i < (*Corr).size(); i++) {
		int s_index, t_index;
		s_index = (*Corr)[i].index_query;
		t_index = (*Corr)[i].index_match;

		if (t_index != -1) {
			float px = Source_Cloud->points[s_index].x;
			float py = Source_Cloud->points[s_index].y;
			float pz = Source_Cloud->points[s_index].z;
			float qx = Target_Cloud->points[t_index].x;
			float qy = Target_Cloud->points[t_index].y;
			float qz = Target_Cloud->points[t_index].z;

			float dx = px - qx;
			float dy = py - qy;
			float dz = pz - qz;

			Eigen::Matrix<double, 3, 6> A_Matrix;
			Eigen::Matrix<double, 3, 1> b_vector;
			Eigen::Matrix<double, 3, 1> residual_vector;

			A_Matrix << 1, 0, 0, 0, pz, -py,
				0, 1, 0, -pz, 0, px,
				0, 0, 1, py, -px, 0;
			b_vector << -dx, -dy, -dz;

			residual_vector = A_Matrix * transform_x - b_vector;

			VTPV += (*Corr)[i].weight * (residual_vector(0) * residual_vector(0) + residual_vector(1) * residual_vector(1) + residual_vector(2) * residual_vector(2));

			observation_count += 3;
		}
	}

	return true;
}

bool MullsCalculate::pt2pl_lls_residual(const pcl::PointCloud<MullsPoint>::Ptr &Source_Cloud, const pcl::PointCloud<MullsPoint>::Ptr &Target_Cloud,
						                   boost::shared_ptr<pcl::Correspondences> &Corr, const Vector6d &transform_x, double &VTPV, int &observation_count) {
	//point-to-plane distance metrics
	//1 observation equation for 1 pair of correspondence
	for (size_t i = 0u; i < (*Corr).size(); i++) {
		int s_index, t_index;
		s_index = (*Corr)[i].index_query;
		t_index = (*Corr)[i].index_match;
		if (t_index != -1) {
			float px = Source_Cloud->points[s_index].x;
			float py = Source_Cloud->points[s_index].y;
			float pz = Source_Cloud->points[s_index].z;
			float qx = Target_Cloud->points[t_index].x;
			float qy = Target_Cloud->points[t_index].y;
			float qz = Target_Cloud->points[t_index].z;
			float ntx = Target_Cloud->points[t_index].normal_x;
			float nty = Target_Cloud->points[t_index].normal_y;
			float ntz = Target_Cloud->points[t_index].normal_z;

			float a = ntz * py - nty * pz;
			float b = ntx * pz - ntz * px;
			float c = nty * px - ntx * py;
			float d = ntx * qx + nty * qy + ntz * qz - ntx * px - nty * py - ntz * pz;

			float residual = ntx * transform_x(0) + nty * transform_x(1) + ntz * transform_x(2) + a * transform_x(3) + b * transform_x(4) + c * transform_x(5) - d;

			VTPV += (*Corr)[i].weight * residual * residual;

			observation_count++;
		}
	}

	return true;
}

bool MullsCalculate::pt2li_lls_residual(const pcl::PointCloud<MullsPoint>::Ptr &Source_Cloud, const pcl::PointCloud<MullsPoint>::Ptr &Target_Cloud,
						                   boost::shared_ptr<pcl::Correspondences> &Corr, const Vector6d &transform_x, double &VTPV, int &observation_count) {
	//point-to-line distance metrics
	//3 observation equation for 1 pair of correspondence
	for (size_t i = 0u; i < (*Corr).size(); i++) {
		int s_index, t_index;
		s_index = (*Corr)[i].index_query;
		t_index = (*Corr)[i].index_match;
		if (t_index != -1) {
			float px = Source_Cloud->points[s_index].x;
			float py = Source_Cloud->points[s_index].y;
			float pz = Source_Cloud->points[s_index].z;
			float qx = Target_Cloud->points[t_index].x;
			float qy = Target_Cloud->points[t_index].y;
			float qz = Target_Cloud->points[t_index].z;
			float vx = Target_Cloud->points[t_index].normal_x; //actually primary directional vector 
			float vy = Target_Cloud->points[t_index].normal_y;
			float vz = Target_Cloud->points[t_index].normal_z;

			float dx = px - qx;
			float dy = py - qy;
			float dz = pz - qz;

			Eigen::Matrix<double, 3, 6> A_Matrix;
			Eigen::Matrix<double, 3, 1> b_vector;
			Eigen::Matrix<double, 3, 1> residual_vector;

			A_Matrix << 0, vz, -vy, -vz * pz - vy * py, vy * px, vz * px,
				-vz, 0, vx, vx * py, -vx * px - vz * pz, vz * py,
				vy, -vx, 0, vx * pz, vy * pz, -vy * py - vx * px;
			b_vector << -vz * dy + vy * dz, -vx * dz + vz * dx, -vy * dx + vx * dy;

			residual_vector = A_Matrix * transform_x - b_vector;

			//LOG(INFO) << "final weight (pt-li): " << (*Corr)[i].weight;

			//VTPV is the sum of square of the residuals
			VTPV += (*Corr)[i].weight * (residual_vector(0) * residual_vector(0) + residual_vector(1) * residual_vector(1) + residual_vector(2) * residual_vector(2));

			observation_count += 3;
		}
	}
	return true;
}

float MullsCalculate::get_weight_by_dist_adaptive(float dist, int iter_num, float unit_dist, float b_min, float b_max, float b_step) {
	float b_current = std::min(b_min + b_step * iter_num, b_max);
	float temp_weight = b_current + (1.0 - b_current) * dist / unit_dist;
	temp_weight = std::max(temp_weight, 0.01f);
	return temp_weight;
}

inline float MullsCalculate::get_weight_by_dist(float dist, float unit_dist, float base_value) {
	return (base_value + (1 - base_value) * dist / unit_dist);
}

inline float MullsCalculate::get_weight_by_intensity(float intensity_1, float intensity_2, float base_value, float intensity_scale) {
	float intensity_diff_ratio = std::fabs(intensity_1 - intensity_2) / intensity_scale;
	float intensity_weight = std::exp(-1.0 * intensity_diff_ratio);
	return intensity_weight;
}

inline float MullsCalculate::get_weight_by_residual(float res, float huber_thre, int delta) {
	//Huber Loss
	//y=0.5*x^2        , x<d
	//y=0.5*d^2+|x|-d  , x>=d
	//d= 1, |x|= res/huber_thre
	//weight=(0.5*d^2+|x|-d)/(0.5*x^2) = (2*res*huber_thre-huber_thre*huber_thre)/res/res)
	return ((res > huber_thre) ? ((2 * res * huber_thre + (delta * delta - 2 * delta) * (huber_thre * huber_thre)) / res / res) : (1.0));
}

float MullsCalculate::get_weight_by_residual_general(float res, float thre, float alpha) {
	float weight;
	res = res / thre;
	if (alpha == 2)
		weight = 1.0;
	else if (alpha == 0)
		weight = 2.0 / (res * res + 2.0);
	else
		weight = 1.0 * std::pow((res * res / std::abs(alpha - 2.0) + 1.0), (alpha * 0.5 - 1.0));

	return weight;
}

bool MullsCalculate::construct_trans_a(const double &tx, const double &ty, const double &tz,
										  const double &alpha, const double &beta, const double &gamma,
										  Eigen::Matrix4d &transformation_matrix) {
	// Construct the transformation matrix from rotation and translation
	transformation_matrix = Eigen::Matrix<double, 4, 4>::Zero();
	// From euler angle to rotation matrix

	transformation_matrix(0, 0) = std::cos(gamma) * std::cos(beta);
	transformation_matrix(0, 1) = -std::sin(gamma) * std::cos(alpha) + std::cos(gamma) * std::sin(beta) * std::sin(alpha);
	transformation_matrix(0, 2) = std::sin(gamma) * std::sin(alpha) + std::cos(gamma) * std::sin(beta) * std::cos(alpha);
	transformation_matrix(1, 0) = std::sin(gamma) * std::cos(beta);
	transformation_matrix(1, 1) = std::cos(gamma) * std::cos(alpha) + std::sin(gamma) * std::sin(beta) * std::sin(alpha);
	transformation_matrix(1, 2) = -std::cos(gamma) * std::sin(alpha) + std::sin(gamma) * std::sin(beta) * std::cos(alpha);
	transformation_matrix(2, 0) = -std::sin(beta);
	transformation_matrix(2, 1) = std::cos(beta) * std::sin(alpha);
	transformation_matrix(2, 2) = std::cos(beta) * std::cos(alpha);

	transformation_matrix(0, 3) = tx;
	transformation_matrix(1, 3) = ty;
	transformation_matrix(2, 3) = tz;
	transformation_matrix(3, 3) = 1.0;

	return true;
}

bool MullsCalculate::get_quat_euler_jacobi(const Eigen::Vector3d &euler_angle, Eigen::Matrix3d &Jacobi) {
	float sin_half_roll, cos_half_roll, sin_half_pitch, cos_half_pitch, sin_half_yaw, cos_half_yaw;

	sin_half_roll = sin(0.5 * euler_angle(0));
	sin_half_pitch = sin(0.5 * euler_angle(1));
	sin_half_yaw = sin(0.5 * euler_angle(2));
	cos_half_roll = cos(0.5 * euler_angle(0));
	cos_half_pitch = cos(0.5 * euler_angle(1));
	cos_half_yaw = cos(0.5 * euler_angle(2));

	//roll pitch yaw (x, y', z'')
	Jacobi(0, 0) = 0.5 * (cos_half_roll * cos_half_pitch * cos_half_yaw + sin_half_roll * sin_half_pitch * sin_half_yaw);
	Jacobi(0, 1) = 0.5 * (-sin_half_roll * sin_half_pitch * cos_half_yaw - cos_half_roll * cos_half_pitch * sin_half_yaw);
	Jacobi(0, 2) = 0.5 * (-sin_half_roll * cos_half_pitch * sin_half_yaw - cos_half_roll * sin_half_pitch * cos_half_yaw);

	Jacobi(1, 0) = 0.5 * (-sin_half_roll * sin_half_pitch * cos_half_yaw + cos_half_roll * cos_half_pitch * sin_half_yaw);
	Jacobi(1, 1) = 0.5 * (cos_half_roll * cos_half_pitch * cos_half_yaw - sin_half_roll * sin_half_pitch * sin_half_yaw);
	Jacobi(1, 2) = 0.5 * (-cos_half_roll * sin_half_pitch * sin_half_yaw + sin_half_roll * cos_half_pitch * cos_half_yaw);

	Jacobi(2, 0) = 0.5 * (-sin_half_roll * cos_half_pitch * sin_half_yaw - cos_half_roll * sin_half_pitch * cos_half_yaw);
	Jacobi(2, 1) = 0.5 * (-cos_half_roll * sin_half_pitch * sin_half_yaw - sin_half_roll * cos_half_pitch * cos_half_yaw);
	Jacobi(2, 2) = 0.5 * (cos_half_roll * cos_half_pitch * cos_half_yaw + sin_half_roll * sin_half_pitch * sin_half_yaw);
	
	return true;
}


bool MullsCalculate::keep_less_source_pts(pcl::PointCloud<MullsPoint>::Ptr &pc_ground_tc,
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
											 int ground_down_rate, int facade_down_rate, int target_down_rate) {
	MullsFilter cfilter;

	cfilter.random_downsample_pcl(pc_ground_tc, (int)(pc_ground_tc->points.size() / target_down_rate));
	cfilter.random_downsample_pcl(pc_facade_tc, (int)(pc_facade_tc->points.size() / target_down_rate));

	cfilter.random_downsample_pcl(pc_ground_sc, (int)(pc_ground_tc->points.size() / ground_down_rate));
	cfilter.random_downsample_pcl(pc_facade_sc, (int)(pc_facade_tc->points.size() / facade_down_rate));
	cfilter.random_downsample_pcl(pc_pillar_sc, (int)(pc_pillar_tc->points.size()));
	cfilter.random_downsample_pcl(pc_beam_sc, (int)(pc_beam_tc->points.size()));
	cfilter.random_downsample_pcl(pc_roof_sc, (int)(pc_roof_tc->points.size()));
	cfilter.random_downsample_pcl(pc_vertex_sc, (int)(pc_vertex_tc->points.size()));
	return true;
}

bool MullsCalculate::intersection_filter(Constraint &registration_cons,
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
											float bbx_pad) {
		MullsFilter cfilter;
		Bounds intersection_bbx, source_init_guess_bbx_merged;
		std::vector<Bounds> source_init_guess_bbxs(3);
		get_cloud_bbx(pc_ground_sc, source_init_guess_bbxs[0]);
		get_cloud_bbx(pc_pillar_sc, source_init_guess_bbxs[1]);
		get_cloud_bbx(pc_facade_sc, source_init_guess_bbxs[2]);
		merge_bbx(source_init_guess_bbxs, source_init_guess_bbx_merged);
		get_intersection_bbx(registration_cons.block1->local_bound, source_init_guess_bbx_merged, intersection_bbx, bbx_pad);
		cfilter.get_cloud_pair_intersection(intersection_bbx,
											pc_ground_tc, pc_pillar_tc, pc_beam_tc, pc_facade_tc, pc_roof_tc, pc_vertex_tc,
											pc_ground_sc, pc_pillar_sc, pc_beam_sc, pc_facade_sc, pc_roof_sc, pc_vertex_sc);
    	return true;
	}

} // namespace mulls
