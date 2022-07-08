#include "cfilter.hpp"
#include "cregistration.hpp"
#include "mulls_util.h"

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
using namespace mulls;

//static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
//#define CUDA_CHECK(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

//GFLAG Template: DEFINE_TYPE(Flag_variable_name, default_value, "Comments")
//file path
DEFINE_string(point_cloud_1_path, "", "Pointcloud 1 file path");
DEFINE_string(point_cloud_2_path, "", "Pointcloud 2 file path");
DEFINE_string(output_point_cloud_path, "", "Registered source pointcloud file path");
DEFINE_string(appro_coordinate_file, "", "File used to store the approximate station coordinate");
//key parameters
DEFINE_double(cloud_1_down_res, 0.0, "voxel size(m) of downsample for target point cloud");
DEFINE_double(cloud_2_down_res, 0.0, "voxel size(m) of downsample for source point cloud");
DEFINE_double(gf_grid_size, 2.0, "grid size(m) of ground segmentation");
DEFINE_double(gf_in_grid_h_thre, 0.3, "height threshold(m) above the lowest point in a grid for ground segmentation");
DEFINE_double(gf_neigh_grid_h_thre, 2.2, "height threshold(m) among neighbor grids for ground segmentation");
DEFINE_double(gf_max_h, DBL_MAX, "max height(m) allowed for ground point");
DEFINE_int32(gf_ground_down_rate, 10, "downsampling decimation rate for target ground point cloud");
DEFINE_int32(gf_nonground_down_rate, 3, "downsampling decimation rate for nonground point cloud");
DEFINE_int32(dist_inverse_sampling_method, 0, "use distance inverse sampling or not (0: disabled, 1: linear weight, 2: quadratic weight)");
DEFINE_double(unit_dist, 15.0, "distance that correspoinding to unit weight in inverse distance downsampling");
DEFINE_bool(pca_distance_adpative_on, false, "enable the distance adpative pca or not. It is preferred to be on if the point cloud is collected by a spining scanner located at origin point");
DEFINE_double(pca_neighbor_radius, 1.0, "pca neighborhood searching radius(m) for target point cloud");
DEFINE_int32(pca_neighbor_count, 30, "use only the k nearest neighbor in the r-neighborhood to do PCA");
DEFINE_double(linearity_thre, 0.6, "pca linearity threshold");
DEFINE_double(planarity_thre, 0.6, "pca planarity threshold");
DEFINE_double(curvature_thre, 0.1, "pca local stability threshold");
DEFINE_int32(corr_num, 3000, "fixed number of the correspondence for global coarse registration (only when fixed_num_corr_on is on).");
DEFINE_bool(reciprocal_corr_on, false, "Using reciprocal correspondence");
DEFINE_bool(fixed_num_corr_on, false, "Using fixed number correspondece (best k matches)");
DEFINE_double(corr_dis_thre, 2.0, "distance threshold between correspondence points");
DEFINE_int32(reg_max_iter_num, 25, "max iteration number for icp-based registration");
DEFINE_double(converge_tran, 0.001, "convergence threshold for translation (in m)");
DEFINE_double(converge_rot_d, 0.01, "convergence threshold for rotation (in degree)");
DEFINE_double(heading_change_step_degree, 15, "The step for the rotation of heading");
DEFINE_bool(is_global_reg, true, "Allow the global registration without good enough initial guess or not");
DEFINE_bool(teaser_on, false, "Using TEASER++ or to do the global coarse registration or not (using RANSAC instead)");
//visualizer parameters
DEFINE_bool(realtime_viewer_on, false, "Launch the real-time registration(correspondence) viewer or not");
DEFINE_int32(screen_width, 1920, "monitor horizontal resolution (pixel)");
DEFINE_int32(screen_height, 1080, "monitor vertical resolution (pixel)");
DEFINE_double(vis_intensity_scale, 256.0, "max intensity value of your data");

void show_register_result(pcl::PointCloud<MullsPoint>::Ptr source_cloud,
                          pcl::PointCloud<MullsPoint>::Ptr target_cloud,
                          std::string name = "show")
{
    // -----------------visualization--------------------------
    // set background color
    std::shared_ptr<pcl::visualization::PCLVisualizer>
        viewer_final(new pcl::visualization::PCLVisualizer(name));
    viewer_final->setBackgroundColor(0, 0, 0);

    // set target color (red)
    pcl::visualization::PointCloudColorHandlerCustom<MullsPoint> target_color(target_cloud, 255, 0, 0);
    viewer_final->addPointCloud<MullsPoint>(target_cloud, target_color, "target cloud");
    viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");

    // set source color (blue)
    pcl::visualization::PointCloudColorHandlerCustom<MullsPoint> source_color(source_cloud, 0, 0, 255);
    viewer_final->addPointCloud<MullsPoint>(source_cloud, source_color, "source cloud");
    viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source cloud");

    while (!viewer_final->wasStopped())
    {
        viewer_final->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
    return;
}

int main(int argc, char **argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging("Mylog_testreg");
    LOG(INFO) << "Launch the program!";
    LOG(INFO) << "Logging is written to " << FLAGS_log_dir;

    CHECK(FLAGS_point_cloud_1_path != "") << "Need to specify the first point cloud.";
    CHECK(FLAGS_point_cloud_2_path != "") << "Need to specify the second point cloud.";
    CHECK(FLAGS_output_point_cloud_path != "") << "Need to specify where to save the registered point cloud.";

    //Import configuration
    std::string filename1 = FLAGS_point_cloud_1_path;      //The first (target) point cloud's file path
    std::string filename2 = FLAGS_point_cloud_2_path;      //The second (source) point cloud's file path
    std::string filenameR = FLAGS_output_point_cloud_path; //Registered source pointcloud file path
    std::string appro_coordinate_file = FLAGS_appro_coordinate_file;

    float vf_downsample_resolution_source = FLAGS_cloud_1_down_res;
    float vf_downsample_resolution_target = FLAGS_cloud_2_down_res;
    float gf_grid_resolution = FLAGS_gf_grid_size;
    float gf_max_grid_height_diff = FLAGS_gf_in_grid_h_thre;
    float gf_neighbor_height_diff = FLAGS_gf_neigh_grid_h_thre;
    float gf_max_height = FLAGS_gf_max_h;
    int ground_down_rate = FLAGS_gf_ground_down_rate;
    int nonground_down_rate = FLAGS_gf_nonground_down_rate;
    int dist_inv_sampling_method = FLAGS_dist_inverse_sampling_method;
    float dist_inv_sampling_dist = FLAGS_unit_dist;
    bool pca_distance_adpative_on = FLAGS_pca_distance_adpative_on;
    float pca_neigh_r = FLAGS_pca_neighbor_radius;
    int pca_neigh_k = FLAGS_pca_neighbor_count;
    float pca_linearity_thre = FLAGS_linearity_thre;
    float pca_planarity_thre = FLAGS_planarity_thre;
    float pca_curvature_thre = FLAGS_curvature_thre;
    int feature_correspondence_num = FLAGS_corr_num;
    float reg_corr_dis_thre = FLAGS_corr_dis_thre;
    float converge_tran = FLAGS_converge_tran;
    float converge_rot_d = FLAGS_converge_rot_d;
    int max_iteration_num = FLAGS_reg_max_iter_num;
    float heading_step_d = FLAGS_heading_change_step_degree;
    bool launch_realtime_viewer = FLAGS_realtime_viewer_on;
    bool global_registration_on = FLAGS_is_global_reg;
    bool teaser_on = FLAGS_teaser_on;
    float pca_linearity_thre_down = pca_linearity_thre + 0.1;
    float pca_planarity_thre_down = pca_planarity_thre + 0.1;
    float keypoint_nms_radius = 0.25 * pca_neigh_r;

    CFilter<MullsPoint> cfilter;
    CRegistration<MullsPoint> creg;

    CloudBlockPtr cblock_1(new CloudBlock()), cblock_2(new CloudBlock());
    cblock_1->filename = filename1;
    cblock_2->filename = filename2;

    // LOAD source and target cloud
    pcl::io::loadPCDFile<MullsPoint>(cblock_1->filename, *cblock_1->pc_raw);
    get_cloud_bbx_cpt(cblock_1->pc_raw, cblock_1->local_bound, cblock_1->local_cp);
    pcl::io::loadPCDFile<MullsPoint>(cblock_2->filename, *cblock_2->pc_raw);
    get_cloud_bbx_cpt(cblock_2->pc_raw, cblock_2->local_bound, cblock_2->local_cp);

    //Extract feature points
    cfilter.extract_semantic_pts(cblock_1, vf_downsample_resolution_target, gf_grid_resolution, gf_max_grid_height_diff,
                                 gf_neighbor_height_diff, gf_max_height, ground_down_rate, nonground_down_rate,
                                 pca_neigh_r, pca_neigh_k, pca_linearity_thre, pca_planarity_thre, pca_curvature_thre,
                                 pca_linearity_thre_down, pca_planarity_thre_down, pca_distance_adpative_on,
                                 dist_inv_sampling_method, dist_inv_sampling_dist);
    cfilter.extract_semantic_pts(cblock_2, vf_downsample_resolution_source, gf_grid_resolution, gf_max_grid_height_diff,
                                 gf_neighbor_height_diff, gf_max_height, ground_down_rate, nonground_down_rate,
                                 pca_neigh_r, pca_neigh_k, pca_linearity_thre, pca_planarity_thre, pca_curvature_thre,
                                 pca_linearity_thre_down, pca_planarity_thre_down, pca_distance_adpative_on,
                                 dist_inv_sampling_method, dist_inv_sampling_dist);

    if (global_registration_on) //refine keypoints
    {
        cfilter.non_max_suppress(cblock_1->pc_vertex, keypoint_nms_radius);
        cfilter.non_max_suppress(cblock_2->pc_vertex, keypoint_nms_radius);
    }

    //Registration
    Constraint reg_con;

    //Assign target (cblock1) and source (cblock2) point cloud for registration
    creg.determine_source_target_cloud(cblock_1, cblock_2, reg_con);

    Eigen::Matrix4d init_mat, ident_mat;
    init_mat.setIdentity();
    ident_mat.setIdentity();
    if (global_registration_on)
    {
        pcTPtr target_cor(new pcT()), source_cor(new pcT());
        creg.find_feature_correspondence_ncc(reg_con.block1->pc_vertex, reg_con.block2->pc_vertex, target_cor, source_cor,
                                             FLAGS_fixed_num_corr_on, feature_correspondence_num, FLAGS_reciprocal_corr_on);
        
        if(teaser_on)
            creg.coarse_reg_teaser(target_cor, source_cor, init_mat, 4.0 * keypoint_nms_radius);
        else
            creg.coarse_reg_ransac(target_cor, source_cor, init_mat, 4.0 * keypoint_nms_radius);
       
    }

    creg.mm_lls_icp(reg_con, max_iteration_num, reg_corr_dis_thre, converge_tran, converge_rot_d, 0.25 * reg_corr_dis_thre,
                        1.1, "111110", "1101", 1.0, 0.1, 0.1, 0.1, init_mat);
    pcTPtr pc_s_tran(new pcT());
    pcl::transformPointCloud(*reg_con.block2->pc_down, *pc_s_tran, reg_con.Trans1_2);
    show_register_result(pc_s_tran, reg_con.block1->pc_down, "mulls");

    return 1;
}
