//
// This file is for all kinds of data I/O
// Dependent 3rd Libs: PCL (>1.7), LibLas (optional for *LAS IO)
// By Yue Pan
//

#ifndef _INCLUDE_DATA_IO_HPP
#define _INCLUDE_DATA_IO_HPP

//pcl
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>

//boost
#include <boost/filesystem.hpp>
#include <boost/function.hpp>

#include <glog/logging.h>
#include <string>
#include <fstream>
#include <vector>

#include "utility.hpp"

using namespace boost::filesystem;
using namespace std;

namespace lo
{
inline std::ostream &operator<<(std::ostream &output, const Eigen::Matrix4d &mat)
{
    output << setprecision(8);

    for (int i = 0; i < 4; i++)
    {
        output << mat(i, 0) << "\t" << mat(i, 1) << "\t" << mat(i, 2) << "\t" << mat(i, 3) << "\n";
    }
    return output;
}

inline std::ostream &operator<<(std::ostream &output, const Matrix6d &mat)
{
    output << setprecision(8);

    for (int i = 0; i < 6; i++)
    {
        output << mat(i, 0) << "\t" << mat(i, 1) << "\t" << mat(i, 2) << "\t"
               << mat(i, 3) << "\t" << mat(i, 4) << "\t" << mat(i, 5) << "\n";
    }
    return output;
}

inline std::istream &operator>>(std::istream &input, pose_qua_t &pose)
{
    input >> pose.trans.x() >> pose.trans.y() >> pose.trans.z() >> pose.quat.x() >>
        pose.quat.y() >> pose.quat.z() >> pose.quat.w();
    // Normalize the quaternion to account for precision loss due to serialization.
    pose.quat.normalize();
    return input;
}

inline std::ostream &operator<<(std::ostream &output, const pose_qua_t &pose)
{
    output << pose.trans.x() << "\t" << pose.trans.y() << "\t" << pose.trans.z() << "\t"
           << pose.quat.x() << "\t" << pose.quat.y() << "\t" << pose.quat.z() << "\t" << pose.quat.w() << "\t";
    return output;
}

inline std::ostream &operator<<(std::ostream &output, const bounds_t &bbx)
{
    output << setprecision(8);
    output << bbx.min_x << "\t" << bbx.min_y << "\t" << bbx.min_z << "\t" << bbx.max_x << "\t" << bbx.max_y << "\t" << bbx.max_z << "\n";
    return output;
}

inline std::ostream &operator<<(std::ostream &output, const centerpoint_t &cp)
{
    output << setprecision(8);
    output << cp.x << "\t" << cp.y << "\t" << cp.z << "\n";
    return output;
}

inline std::ostream &operator<<(std::ostream &output, const constraint_t &con)
{
    output << con.unique_id << "\t" << con.con_type << "\t"
           << con.block1->unique_id << "\t" << con.block1->data_type << "\t"
           << con.block2->unique_id << "\t" << con.block2->data_type << "\n";

    output << con.Trans1_2;

    output << con.information_matrix;

    return output;
}

template <typename PointT>
class DataIo : public CloudUtility<PointT>
{
  public:
    bool check_dir(const std::string &dir)
    {
        if (!boost::filesystem::exists(dir.c_str()))
        {
            if (boost::filesystem::create_directory(dir.c_str()))
                return true;
            else
                return false;
        }
        return true;
    }

    inline bool exists_file(const std::string &filename)
    {
        std::ifstream f(filename.c_str());
        return f.good();
    }

    // Delete invalid characters such as spaces, tabs, etc. in strings
    std::string trim_str(std::string &str)
    {
        str.erase(0, str.find_first_not_of(" \t\r\n"));
        str.erase(str.find_last_not_of(" \t\r\n") + 1);
        return str;
    }

    bool read_pc_cloud_block(cloudblock_Ptr &in_block, bool normalize_intensity_or_not = false)
    {
        if (read_cloud_file(in_block->filename, in_block->pc_raw))
        {
            this->get_cloud_bbx_cpt(in_block->pc_raw, in_block->local_bound, in_block->local_cp);

            if (normalize_intensity_or_not)
            {
                float min_intensity = FLT_MAX;
                float max_intensity = -FLT_MAX;
                for (int i = 0; i < in_block->pc_raw->points.size(); i++)
                {
                    min_intensity = min_(min_intensity, in_block->pc_raw->points[i].intensity);
                    max_intensity = max_(max_intensity, in_block->pc_raw->points[i].intensity);
                }
                float intesnity_scale = 255.0 / (max_intensity - min_intensity); //rescale to 0-255
                for (int i = 0; i < in_block->pc_raw->points.size(); i++)
                    in_block->pc_raw->points[i].intensity = (in_block->pc_raw->points[i].intensity - min_intensity) * intesnity_scale;
            }

            return 1;
        }
        else
            return 0;
    }

    //Brief: Read the point cloud data of various format
    bool read_cloud_file(const std::string &fileName, typename pcl::PointCloud<PointT>::Ptr &pointCloud)
    {
        std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();

        std::string extension;
        extension = fileName.substr(fileName.find_last_of('.') + 1); //Get the suffix of the file;

        if (!strcmp(extension.c_str(), "pcd"))
        {
            read_pcd_file(fileName, pointCloud);
            LOG(INFO) << "A pcd file has been imported.";
            //std::cout << "A pcd file has been imported" << std::endl;
        }

        std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);

        LOG(INFO) << "[" << pointCloud->points.size() << "] points loaded in [" << time_used.count() * 1000 << "] ms";
        //std::cout << "[" << pointCloud->points.size() << "] points loaded in [" << time_used.count() << "] s" << std::endl;

        return 1;
    }

    bool write_cloud_file(const std::string &fileName, const typename pcl::PointCloud<PointT>::Ptr &pointCloud)
    {
        std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();

        std::string extension;
        extension = fileName.substr(fileName.find_last_of('.') + 1); //Get the suffix of the file;

        if (!strcmp(extension.c_str(), "pcd"))
        {
            write_pcd_file(fileName, pointCloud); //write out ascii format so that CC can load the file
            LOG(INFO) << "A pcd file has been exported" << std::endl;
        }

        std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);

        LOG(INFO) << "[" << pointCloud->points.size() << "] points exported in [" << time_used.count() * 1000 << "] ms";
        return true;
    }

    bool read_pcd_file(const std::string &fileName, typename pcl::PointCloud<PointT>::Ptr &pointCloud)
    {
        if (pcl::io::loadPCDFile<PointT>(fileName, *pointCloud) == -1)
        {
            PCL_ERROR("Couldn't read file\n");
            return false;
        }
        return true;
    }

    bool write_pcd_file(const std::string &fileName, const typename pcl::PointCloud<PointT>::Ptr &pointCloud, bool as_binary = true)
    {
        //do the reshaping
        pointCloud->width = 1;
        pointCloud->height = pointCloud->points.size();

        if (as_binary)
        {
            if (pcl::io::savePCDFileBinary(fileName, *pointCloud) == -1)
            {
                PCL_ERROR("Couldn't write file\n");
                return false;
            }
        }
        else
        {
            if (pcl::io::savePCDFile(fileName, *pointCloud) == -1)
            {
                PCL_ERROR("Couldn't write file\n");
                return false;
            }
        }
        return true;
    }
};
} // namespace lo

#endif // _INCLUDE_DATA_IO_HPP