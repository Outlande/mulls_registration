#include "mulls_filter.h"
#include "mulls_calculate.h"
#include "mulls_util.h"
#include "mulls_registration.h"

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
using namespace mapping_framework::common;

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
    //Import configuration
    std::string filename1 = "/home/levi/Desktop/ali/demo_data/pcd/000000.pcd";      //The first (target) point cloud's file path
    std::string filename2 = "/home/levi/Desktop/ali/demo_data/pcd/000005.pcd";      //The second (source) point cloud's file path

    // LOAD source and target cloud
    pcl::PointCloud<MullsPoint>::Ptr target_cloud(new pcl::PointCloud<MullsPoint>);
    pcl::PointCloud<MullsPoint>::Ptr source_cloud(new pcl::PointCloud<MullsPoint>);
    pcl::io::loadPCDFile<MullsPoint>(filename1, *target_cloud);
    pcl::io::loadPCDFile<MullsPoint>(filename2, *source_cloud);
    

    MullsRegistration* regis(new MullsRegistration);
    regis->SetSourceCloud(source_cloud);
    regis->SetTargetCloud(target_cloud);
    regis->Align(Eigen::Matrix4d::Identity());
    std::cout << regis->final_odom_pose_ << std::endl;

    pcl::PointCloud<MullsPoint>::Ptr source_tran(new pcl::PointCloud<MullsPoint>());
    pcl::transformPointCloud(*regis->source_mulls_->pc_down, *source_tran, regis->final_odom_pose_);
    show_register_result(source_tran, regis->target_mulls_->pc_down, "mulls");

    return 1;
}
