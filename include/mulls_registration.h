#ifndef _INCLUDE_MULLS_REG_H
#define _INCLUDE_MULLS_REG_H

#include "mulls_util.h"
#include "mulls_filter.h"
#include "mulls_calculate.h"

namespace mulls {
class MullsRegistration
{

public:
    MullsRegistration();

    ~MullsRegistration();

    void SetSourceCloud(pcl::PointCloud<MullsPoint>::Ptr source_cloud);

    void SetTargetCloud(pcl::PointCloud<MullsPoint>::Ptr target_cloud);

    void Align(Eigen::Matrix4d init_pose);

    Eigen::Matrix4d final_odom_pose_;

    CloudBlockPtr source_mulls_;

    CloudBlockPtr target_mulls_;

private:
    MullsFilter mulls_filter_;

    MullsCalculate mulls_cal_;
};
}

#endif //_INCLUDE_MULLS_REG_H