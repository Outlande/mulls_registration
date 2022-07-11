#include "mulls_registration.h"

namespace mulls {
float kGfGridResolution = 2.0; // 2.5
float kGfMaxGridHeightDiff = 0.25;
float kGfNeighborHeightDiff = 1.2; // 1.5
float kGfMaxGroundHeight = DBL_MAX; // 2.0
int kGfDownRateGround = 10; // 12
int kGfDownsampleRateNonground = 3;
float kPcaReighborRadius = 1.0; // 0.6
int kPcaNeighborK = 50; //25
float kEdgeThre = 0.65;
float kPlanarThre = 0.65;
float kCurvatureThre = 0.1;
float kEdgeThreDown = 0.75;
float kPlanarThreDown = 0.75;
bool kUseDistanceAdaptivePca = false; // true
int kDistanceInverseSamplingMethod = 2;
float kStandardDistance = 15.0; //10
float KConvergeTranMeter = 0.001;
float KConvergeRatDegree = 0.01;


int max_iteration = 10;
float corr_dist_meter = 3;

MullsRegistration::MullsRegistration( ) {
    source_mulls_.reset(new CloudBlock());
    target_mulls_.reset(new CloudBlock());
}

MullsRegistration::~MullsRegistration()
{}

void MullsRegistration::SetSourceCloud(pcl::PointCloud<MullsPoint>::Ptr source_cloud) {
    source_mulls_.reset(new CloudBlock());
    pcl::copyPointCloud(*source_cloud, *source_mulls_->pc_raw);
    get_cloud_bbx_cpt(source_mulls_->pc_raw, source_mulls_->local_bound, source_mulls_->local_cp);
}

void MullsRegistration::SetTargetCloud(pcl::PointCloud<MullsPoint>::Ptr target_cloud) {
    target_mulls_.reset(new CloudBlock());
    pcl::copyPointCloud(*target_cloud, *target_mulls_->pc_raw);
    get_cloud_bbx_cpt(target_mulls_->pc_raw, target_mulls_->local_bound, target_mulls_->local_cp);
}

void MullsRegistration::Align(Eigen::Matrix4d init_pose) {
    //Extract feature points
    bool global_registration_on = true;
    mulls_filter_.extract_semantic_pts(target_mulls_, kGfGridResolution, kGfMaxGridHeightDiff,
                                       kGfNeighborHeightDiff, kGfMaxGroundHeight, kGfDownRateGround,
                                       kGfDownsampleRateNonground, kPcaReighborRadius, kPcaNeighborK,
                                       kEdgeThre, kPlanarThre, kCurvatureThre, kEdgeThreDown, 
                                       kPlanarThreDown, kUseDistanceAdaptivePca, kDistanceInverseSamplingMethod, 
                                       kStandardDistance);

    mulls_filter_.extract_semantic_pts(source_mulls_, kGfGridResolution, kGfMaxGridHeightDiff,
                                       kGfNeighborHeightDiff, kGfMaxGroundHeight, kGfDownRateGround,
                                       kGfDownsampleRateNonground, kPcaReighborRadius, kPcaNeighborK,
                                       kEdgeThre, kPlanarThre, kCurvatureThre, kEdgeThreDown, 
                                       kPlanarThreDown, kUseDistanceAdaptivePca, kDistanceInverseSamplingMethod, 
                                       kStandardDistance);
    
    //Assign target and source point cloud for registration
    Constraint reg_con;
    mulls_cal_.determine_target_source_cloud(target_mulls_, source_mulls_, reg_con);

    Eigen::Matrix4d init_mat = init_pose;
    // if global_registration_on, use ransec to calculate the initial pose and feature match
    if (global_registration_on) {
        float keypoint_nms_radius = 0.25 * kPcaReighborRadius;
        //refine keypoints
        mulls_filter_.non_max_suppress(target_mulls_->pc_vertex, keypoint_nms_radius);
        mulls_filter_.non_max_suppress(source_mulls_->pc_vertex, keypoint_nms_radius);

        pcl::PointCloud<MullsPoint>::Ptr target_cor(new pcl::PointCloud<MullsPoint>());
        pcl::PointCloud<MullsPoint>::Ptr source_cor(new pcl::PointCloud<MullsPoint>());

        mulls_cal_.find_feature_correspondence_ncc(reg_con.block1->pc_vertex, reg_con.block2->pc_vertex, target_cor, source_cor,
                                                   false, 3000, false);
        mulls_cal_.coarse_reg_ransac(target_cor, source_cor, init_mat, 4.0 * keypoint_nms_radius);
    }

    mulls_cal_.mm_lls_icp(reg_con, max_iteration, corr_dist_meter, KConvergeTranMeter, KConvergeRatDegree, 0.25 * corr_dist_meter,
                          1.1, "111110", "1101", 1.0, 0.1, 0.1, 0.1, init_mat);
    final_odom_pose_ = reg_con.Trans1_2;
}

}
