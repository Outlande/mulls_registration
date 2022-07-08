//
// This file is a General Definition and Tool for Point Cloud Processing based on PCL.
// Dependent 3rd Libs: PCL (>1.7)
// By Yue Pan et al.
//
#ifndef _INCLUDE_UTILITY_HPP_
#define _INCLUDE_UTILITY_HPP_

//PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/point_representation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/transformation_estimation_svd.h>

//Eigen
#include <Eigen/Core>

#include <vector>
#include <list>
#include <chrono>
#include <limits>
#include <time.h>

#include <glog/logging.h>

using namespace std;

//TypeDef
//Select from these two (with/without intensity)
//mind that 'curvature' here is used as ring number for spining scanner
typedef pcl::PointXYZINormal MullsPoint;

/**
//pcl::PointXYZINormal member variables
//x,y,z: 3d coordinates
//intensity: relective intensity
//normal_x, normal_y, normal_z: used as normal or direction vector
//curvature: used as timestamp or descriptor 
//data[3]: used as neighborhood curvature
//normal[3]: used as the height above ground
**/
typedef pcl::PointCloud<MullsPoint>::Ptr pcTPtr;
typedef pcl::PointCloud<MullsPoint> pcT;

typedef pcl::search::KdTree<MullsPoint>::Ptr pcTreePtr;
typedef pcl::search::KdTree<MullsPoint> pcTree;

typedef pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhPtr;
typedef pcl::PointCloud<pcl::FPFHSignature33> fpfh;

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

namespace lo
{

struct CenterPoint {
	double x;
	double y;
	double z;
	CenterPoint(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
};

//regular bounding box whose edges are parallel to x,y,z axises
struct Bounds {
	double min_x;
	double min_y;
	double min_z;
	double max_x;
	double max_y;
	double max_z;
	int type;

	Bounds() {
		min_x = min_y = min_z = max_x = max_y = max_z = 0.0;
	}
	void inf_x() {
		min_x = -DBL_MAX;
		max_x = DBL_MAX;
	}
	void inf_y() {
		min_y = -DBL_MAX;
		max_y = DBL_MAX;
	}
	void inf_z() {
		min_z = -DBL_MAX;
		max_z = DBL_MAX;
	}
	void inf_xyz() {
		inf_x();
		inf_y();
		inf_z();
	}
};

//Basic processing unit(node)
struct CloudBlock
{
	//Strip (transaction) should be the container of the cloudblock while cloudblock can act as either a frame or submap (local map)
	int unique_id;		  //Unique ID
	int strip_id;		  //Strip ID
	int id_in_strip;	  //ID in the strip
	int last_frame_index; //last_frame_id is the frame index (not unique_id) of the last frame of the submap
	//ID means the number may not be continous and begining from 0 (like 3, 7, 11, ...),
	//but index should begin from 0 and its step (interval) should be 1 (like 0, 1, 2, 3, ...)

	Bounds bound;				  //Bounding Box in geo-coordinate system
	CenterPoint cp;			  //Center Point in geo-coordinate system
	CenterPoint station;		  //Station position in geo-coordinate system
	Eigen::Matrix4d station_pose; //Station pose in geo-coordinate system

	Bounds local_bound;				//Bounding Box in local coordinate system
	CenterPoint local_cp;				//Center Point in local coordinate system
	CenterPoint local_station;		//Station position in local coordinate system
	Eigen::Matrix4d local_station_pose; //Station pose in local coordinate system

	bool station_position_available; //If the approximate position of the station is provided
	bool station_pose_available;	 //If the approximate pose of the station is provided
	bool is_single_scanline;		 //If the scanner is a single scanline sensor, determining if adjacent cloud blocks in a strip would have overlap

	bool pose_fixed = false;  //the pose is fixed or not
	bool pose_stable = false; //the pose is stable or not after the optimization

	//poses
	Eigen::Matrix4d pose_lo;		//used for lidar odometry
	Eigen::Matrix4d pose_gt;		//used for lidar odometry (ground turth)
	Eigen::Matrix4d pose_optimized; //optimized pose
	Eigen::Matrix4d pose_init;		//used for the init guess for pgo

	Matrix6d information_matrix_to_next;

	std::string filename;			//full path of the original point cloud file
	std::string filenmae_processed; //full path of the processed point cloud file

	//Raw point cloud
	pcTPtr pc_raw;

	//Downsampled point cloud
	pcTPtr pc_down;
	pcTPtr pc_sketch; //very sparse point cloud

	pcTPtr pc_raw_w; //in world coordinate system (for lidar odometry)

	//unground point cloud
	pcTPtr pc_unground;

	// All kinds of geometric feature points (in target scan)
	pcTPtr pc_ground;
	pcTPtr pc_facade;
	pcTPtr pc_roof;
	pcTPtr pc_pillar;
	pcTPtr pc_beam;
	pcTPtr pc_vertex;
    
	//downsampled feature points (in source scan)
	pcTPtr pc_ground_down;
	pcTPtr pc_facade_down;
	pcTPtr pc_roof_down;
	pcTPtr pc_pillar_down;
	pcTPtr pc_beam_down;

	//Kdtree of the feature points (denser ones)
	pcTreePtr tree_ground;
	pcTreePtr tree_pillar;
	pcTreePtr tree_beam;
	pcTreePtr tree_facade;
	pcTreePtr tree_roof;
	pcTreePtr tree_vertex;

	//actually, it's better to save the indices of feature_points_down instead of saving another feature point cloud

	int down_feature_point_num;
	int feature_point_num;

	CloudBlock()
	{
		init();
		//default value
		station_position_available = false;
		station_pose_available = false;
		is_single_scanline = true;
		pose_lo.setIdentity();
		pose_gt.setIdentity();
		pose_optimized.setIdentity();
		pose_init.setIdentity();
		information_matrix_to_next.setIdentity();
	}

	CloudBlock(const CloudBlock &in_block, bool clone_feature = false, bool clone_raw = false)
	{
		init();
		clone_metadata(in_block);

		if (clone_feature)
		{
			//clone point cloud (instead of pointer)
			*pc_ground = *(in_block.pc_ground);
			*pc_pillar = *(in_block.pc_pillar);
			*pc_facade = *(in_block.pc_facade);
			*pc_beam = *(in_block.pc_beam);
			*pc_roof = *(in_block.pc_roof);
			*pc_vertex = *(in_block.pc_vertex);
			// keypoint_bsc = in_block.keypoint_bsc;
		}
		if (clone_raw)
			*pc_raw = *(in_block.pc_raw);
	}

	void init()
	{
		pc_raw = boost::make_shared<pcT>();
		pc_down = boost::make_shared<pcT>();
		pc_raw_w = boost::make_shared<pcT>();
		pc_sketch = boost::make_shared<pcT>();
		pc_unground = boost::make_shared<pcT>();

		pc_ground = boost::make_shared<pcT>();
		pc_facade = boost::make_shared<pcT>();
		pc_roof = boost::make_shared<pcT>();
		pc_pillar = boost::make_shared<pcT>();
		pc_beam = boost::make_shared<pcT>();
		pc_vertex = boost::make_shared<pcT>();

		pc_ground_down = boost::make_shared<pcT>();
		pc_facade_down = boost::make_shared<pcT>();
		pc_roof_down = boost::make_shared<pcT>();
		pc_pillar_down = boost::make_shared<pcT>();
		pc_beam_down = boost::make_shared<pcT>();

		init_tree();

		//doubleVectorSBF().swap(keypoint_bsc);

		down_feature_point_num = 0;
		feature_point_num = 0;
	}

	void init_tree()
	{
		tree_ground = boost::make_shared<pcTree>();
		tree_facade = boost::make_shared<pcTree>();
		tree_pillar = boost::make_shared<pcTree>();
		tree_beam = boost::make_shared<pcTree>();
		tree_roof = boost::make_shared<pcTree>();
		tree_vertex = boost::make_shared<pcTree>();
	}

	void free_raw_cloud()
	{
		pc_raw.reset(new pcT());
		pc_down.reset(new pcT());
		pc_unground.reset(new pcT());
	}

	void free_tree()
	{
		tree_ground.reset(new pcTree());
		tree_facade.reset(new pcTree());
		tree_pillar.reset(new pcTree());
		tree_beam.reset(new pcTree());
		tree_roof.reset(new pcTree());
		tree_vertex.reset(new pcTree());
	}

	void free_all()
	{
		free_raw_cloud();
		free_tree();
		pc_ground.reset(new pcT());
		pc_facade.reset(new pcT());
		pc_pillar.reset(new pcT());
		pc_beam.reset(new pcT());
		pc_roof.reset(new pcT());
		pc_vertex.reset(new pcT());
		pc_ground_down.reset(new pcT());
		pc_facade_down.reset(new pcT());
		pc_pillar_down.reset(new pcT());
		pc_beam_down.reset(new pcT());
		pc_roof_down.reset(new pcT());
		pc_sketch.reset(new pcT());
		pc_raw_w.reset(new pcT());
		//doubleVectorSBF().swap(keypoint_bsc);
	}

	void clone_metadata(const CloudBlock &in_cblock)
	{
		feature_point_num = in_cblock.feature_point_num;
		bound = in_cblock.bound;
		local_bound = in_cblock.local_bound;
		local_cp = in_cblock.local_cp;
		pose_lo = in_cblock.pose_lo;
		pose_gt = in_cblock.pose_gt;
		pose_init = in_cblock.pose_init;
		pose_optimized = in_cblock.pose_optimized;
		unique_id = in_cblock.unique_id;
		id_in_strip = in_cblock.id_in_strip;
		filename = in_cblock.filename;
	}

	void append_feature(const CloudBlock &in_cblock, bool append_down, std::string used_feature_type)
	{
		//pc_raw->points.insert(pc_raw->points.end(), in_cblock.pc_raw->points.begin(), in_cblock.pc_raw->points.end());
		if (!append_down)
		{
			if (used_feature_type[0] == '1')
				pc_ground->points.insert(pc_ground->points.end(), in_cblock.pc_ground->points.begin(), in_cblock.pc_ground->points.end());
			if (used_feature_type[1] == '1')
				pc_pillar->points.insert(pc_pillar->points.end(), in_cblock.pc_pillar->points.begin(), in_cblock.pc_pillar->points.end());
			if (used_feature_type[2] == '1')
				pc_facade->points.insert(pc_facade->points.end(), in_cblock.pc_facade->points.begin(), in_cblock.pc_facade->points.end());
			if (used_feature_type[3] == '1')
				pc_beam->points.insert(pc_beam->points.end(), in_cblock.pc_beam->points.begin(), in_cblock.pc_beam->points.end());
			if (used_feature_type[4] == '1')
				pc_roof->points.insert(pc_roof->points.end(), in_cblock.pc_roof->points.begin(), in_cblock.pc_roof->points.end());
			pc_vertex->points.insert(pc_vertex->points.end(), in_cblock.pc_vertex->points.begin(), in_cblock.pc_vertex->points.end());
		}
		else
		{
			if (used_feature_type[0] == '1')
				pc_ground->points.insert(pc_ground->points.end(), in_cblock.pc_ground_down->points.begin(), in_cblock.pc_ground_down->points.end());
			if (used_feature_type[1] == '1')
				pc_pillar->points.insert(pc_pillar->points.end(), in_cblock.pc_pillar_down->points.begin(), in_cblock.pc_pillar_down->points.end());
			if (used_feature_type[2] == '1')
				pc_facade->points.insert(pc_facade->points.end(), in_cblock.pc_facade_down->points.begin(), in_cblock.pc_facade_down->points.end());
			if (used_feature_type[3] == '1')
				pc_beam->points.insert(pc_beam->points.end(), in_cblock.pc_beam_down->points.begin(), in_cblock.pc_beam_down->points.end());
			if (used_feature_type[4] == '1')
				pc_roof->points.insert(pc_roof->points.end(), in_cblock.pc_roof_down->points.begin(), in_cblock.pc_roof_down->points.end());
			pc_vertex->points.insert(pc_vertex->points.end(), in_cblock.pc_vertex->points.begin(), in_cblock.pc_vertex->points.end());
		}
	}

	void merge_feature_points(pcTPtr &pc_out, bool merge_down, bool with_out_ground = false)
	{
		if (!merge_down)
		{
			if (!with_out_ground)
				pc_out->points.insert(pc_out->points.end(), pc_ground->points.begin(), pc_ground->points.end());
			pc_out->points.insert(pc_out->points.end(), pc_facade->points.begin(), pc_facade->points.end());
			pc_out->points.insert(pc_out->points.end(), pc_pillar->points.begin(), pc_pillar->points.end());
			pc_out->points.insert(pc_out->points.end(), pc_beam->points.begin(), pc_beam->points.end());
			pc_out->points.insert(pc_out->points.end(), pc_roof->points.begin(), pc_roof->points.end());
			pc_out->points.insert(pc_out->points.end(), pc_vertex->points.begin(), pc_vertex->points.end());
		}
		else
		{
			if (!with_out_ground)
				pc_out->points.insert(pc_out->points.end(), pc_ground_down->points.begin(), pc_ground_down->points.end());
			pc_out->points.insert(pc_out->points.end(), pc_facade_down->points.begin(), pc_facade_down->points.end());
			pc_out->points.insert(pc_out->points.end(), pc_pillar_down->points.begin(), pc_pillar_down->points.end());
			pc_out->points.insert(pc_out->points.end(), pc_beam_down->points.begin(), pc_beam_down->points.end());
			pc_out->points.insert(pc_out->points.end(), pc_roof_down->points.begin(), pc_roof_down->points.end());
			pc_out->points.insert(pc_out->points.end(), pc_vertex->points.begin(), pc_vertex->points.end());
		}
	}

	void transform_feature(const Eigen::Matrix4d &trans_mat, bool transform_down = true, bool transform_undown = true)
	{
		if (transform_undown)
		{
			pcl::transformPointCloudWithNormals(*pc_ground, *pc_ground, trans_mat);
			pcl::transformPointCloudWithNormals(*pc_pillar, *pc_pillar, trans_mat);
			pcl::transformPointCloudWithNormals(*pc_beam, *pc_beam, trans_mat);
			pcl::transformPointCloudWithNormals(*pc_facade, *pc_facade, trans_mat);
			pcl::transformPointCloudWithNormals(*pc_roof, *pc_roof, trans_mat);
			pcl::transformPointCloudWithNormals(*pc_vertex, *pc_vertex, trans_mat);
		}
		if (transform_down)
		{
			pcl::transformPointCloudWithNormals(*pc_ground_down, *pc_ground_down, trans_mat);
			pcl::transformPointCloudWithNormals(*pc_pillar_down, *pc_pillar_down, trans_mat);
			pcl::transformPointCloudWithNormals(*pc_beam_down, *pc_beam_down, trans_mat);
			pcl::transformPointCloudWithNormals(*pc_facade_down, *pc_facade_down, trans_mat);
			pcl::transformPointCloudWithNormals(*pc_roof_down, *pc_roof_down, trans_mat);
		}
	}

	void clone_cloud(pcTPtr &pc_out, bool get_pc_done)
	{
		if (get_pc_done)
			pc_out->points.insert(pc_out->points.end(), pc_down->points.begin(), pc_down->points.end());
		else
			pc_out->points.insert(pc_out->points.end(), pc_raw->points.begin(), pc_raw->points.end());
	}

	void clone_feature(pcTPtr &pc_ground_out,
					   pcTPtr &pc_pillar_out,
					   pcTPtr &pc_beam_out,
					   pcTPtr &pc_facade_out,
					   pcTPtr &pc_roof_out,
					   pcTPtr &pc_vertex_out, bool get_feature_down)

	{
		if (get_feature_down)
		{
			*pc_ground_out = *pc_ground_down;
			*pc_pillar_out = *pc_pillar_down;
			*pc_beam_out = *pc_beam_down;
			*pc_facade_out = *pc_facade_down;
			*pc_roof_out = *pc_roof_down;
			*pc_vertex_out = *pc_vertex;
		}
		else
		{
			*pc_ground_out = *pc_ground;
			*pc_pillar_out = *pc_pillar;
			*pc_beam_out = *pc_beam;
			*pc_facade_out = *pc_facade;
			*pc_roof_out = *pc_roof;
			*pc_vertex_out = *pc_vertex;
		}
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef std::vector<CloudBlock, Eigen::aligned_allocator<CloudBlock>> strip;
typedef std::vector<strip> strips;
typedef boost::shared_ptr<CloudBlock> CloudBlockPtr;
typedef std::vector<CloudBlockPtr> CloudBlockPtrs;

//the edge of pose(factor) graph
struct Constraint
{
	int unique_id;				   //Unique ID
	CloudBlockPtr block1, block2; //Two block  //Target: block1,  Source: block2
	Eigen::Matrix4d Trans1_2;	  //transformation from 2 to 1 (in global shifted map coordinate system)
	Matrix6d information_matrix;
	float overlapping_ratio; //overlapping ratio (not bbx IOU) of two cloud blocks
	float confidence;
	float sigma;			  //standard deviation of the edge
	bool cov_updated = false; //has the information_matrix already updated

	Constraint()
	{
		block1 = CloudBlockPtr(new CloudBlock);
		block2 = CloudBlockPtr(new CloudBlock);
		Trans1_2.setIdentity();
		information_matrix.setIdentity();
		sigma = FLT_MAX;
		cov_updated = false;
	}

	void free_cloud()
	{
		block1->free_all();
		block2->free_all();
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

//Get Bound of a Point Cloud
void get_cloud_bbx(const pcl::PointCloud<MullsPoint>::Ptr &cloud, Bounds &bound)
{
	double min_x = DBL_MAX;
	double min_y = DBL_MAX;
	double min_z = DBL_MAX;
	double max_x = -DBL_MAX;
	double max_y = -DBL_MAX;
	double max_z = -DBL_MAX;

	for (int i = 0; i < cloud->points.size(); i++)
	{
		if (min_x > cloud->points[i].x)
			min_x = cloud->points[i].x;
		if (min_y > cloud->points[i].y)
			min_y = cloud->points[i].y;
		if (min_z > cloud->points[i].z)
			min_z = cloud->points[i].z;
		if (max_x < cloud->points[i].x)
			max_x = cloud->points[i].x;
		if (max_y < cloud->points[i].y)
			max_y = cloud->points[i].y;
		if (max_z < cloud->points[i].z)
			max_z = cloud->points[i].z;
	}
	bound.min_x = min_x;
	bound.max_x = max_x;
	bound.min_y = min_y;
	bound.max_y = max_y;
	bound.min_z = min_z;
	bound.max_z = max_z;
}

void get_cloud_bbx_cpt(const pcl::PointCloud<MullsPoint>::Ptr &cloud, Bounds &bound, CenterPoint &cp)
{
	get_cloud_bbx(cloud, bound);
	cp.x = 0.5 * (bound.min_x + bound.max_x);
	cp.y = 0.5 * (bound.min_y + bound.max_y);
	cp.z = 0.5 * (bound.min_z + bound.max_z);
}

//Get Center of a Point Cloud
void get_cloud_cpt(const typename pcl::PointCloud<MullsPoint>::Ptr &cloud, CenterPoint &cp)
{
	double cx = 0, cy = 0, cz = 0;
	int point_num = cloud->points.size();

	for (int i = 0; i < point_num; i++)
	{
		cx += cloud->points[i].x / point_num;
		cy += cloud->points[i].y / point_num;
		cz += cloud->points[i].z / point_num;
	}
	cp.x = cx;
	cp.y = cy;
	cp.z = cz;
}

void get_intersection_bbx(Bounds &bbx_1, Bounds &bbx_2, Bounds &bbx_intersection, float bbx_boundary_pad = 2.0)
{
	bbx_intersection.min_x = std::max(bbx_1.min_x, bbx_2.min_x) - bbx_boundary_pad;
	bbx_intersection.min_y = std::max(bbx_1.min_y, bbx_2.min_y) - bbx_boundary_pad;
	bbx_intersection.min_z = std::max(bbx_1.min_z, bbx_2.min_z) - bbx_boundary_pad;
	bbx_intersection.max_x = std::min(bbx_1.max_x, bbx_2.max_x) + bbx_boundary_pad;
	bbx_intersection.max_y = std::min(bbx_1.max_y, bbx_2.max_y) + bbx_boundary_pad;
	bbx_intersection.max_z = std::min(bbx_1.max_z, bbx_2.max_z) + bbx_boundary_pad;
}

void merge_bbx(std::vector<Bounds> &bbxs, Bounds &bbx_merged)
{
	bbx_merged.min_x = DBL_MAX;
	bbx_merged.min_y = DBL_MAX;
	bbx_merged.min_z = DBL_MAX;
	bbx_merged.max_x = -DBL_MAX;
	bbx_merged.max_y = -DBL_MAX;
	bbx_merged.max_z = -DBL_MAX;

	for (int i = 0; i < bbxs.size(); i++)
	{
		bbx_merged.min_x = std::min(bbx_merged.min_x, bbxs[i].min_x);
		bbx_merged.min_y = std::min(bbx_merged.min_y, bbxs[i].min_y);
		bbx_merged.min_z = std::min(bbx_merged.min_z, bbxs[i].min_z);
		bbx_merged.max_x = std::max(bbx_merged.max_x, bbxs[i].max_x);
		bbx_merged.max_y = std::max(bbx_merged.max_y, bbxs[i].max_y);
		bbx_merged.max_z = std::max(bbx_merged.max_z, bbxs[i].max_z);
	}
}

} // namespace lo

//TODO: reproduce the code with better data structure

#endif //_INCLUDE_UTILITY_HPP_