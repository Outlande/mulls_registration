
#ifndef _INCLUDE_MULLS_UTIL_
#define _INCLUDE_MULLS_UTIL_

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

//TypeDef
//Select from these two (with/without intensity)
//mind that 'curvature' here is used as ring number for spining scanner
typedef pcl::PointXYZINormal MullsPoint;

typedef pcl::PointCloud<MullsPoint>::Ptr pcTPtr;
typedef pcl::PointCloud<MullsPoint> pcT;

typedef pcl::search::KdTree<MullsPoint>::Ptr pcTreePtr;
typedef pcl::search::KdTree<MullsPoint> pcTree;

typedef pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhPtr;
typedef pcl::PointCloud<pcl::FPFHSignature33> fpfh;

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

namespace mulls
{

enum TransformEstimationType
{
	SVD,
	LM,
	LLS
};

enum CorresEstimationType
{
	NN,
	NS
}; //NN: Nearest Neighbor ; NS: Normal Shooting

enum DistMetricType
{
	Point2Point,
	Point2Plane,
	Plane2Plane
};

// Eigen Value ,lamada1 > lamada2 > lamada3;
struct eigenvalue_t {
	double lamada1;
	double lamada2;
	double lamada3;
};

//the eigen vector corresponding to the eigen value
struct eigenvector_t {
	Eigen::Vector3f principalDirection;
	Eigen::Vector3f middleDirection;
	Eigen::Vector3f normalDirection;
};

// PCA results
struct pca_feature_t {
	eigenvalue_t values;
	eigenvector_t vectors;
	double curvature;
	double linear;
	double planar;
	double spherical;
	double linear_2;
	double planar_2;
	double spherical_2;
	double normal_diff_ang_deg;
	pcl::PointNormal pt;
	int ptId;
	int pt_num = 0;
	std::vector<int> neighbor_indices;
	std::vector<bool> close_to_query_point;
};

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

struct idpair_t {
	int idx;
	unsigned long long voxel_idx;
	idpair_t() : idx(0), voxel_idx(0u) {}		
	bool operator<(const idpair_t &pair) { return voxel_idx < pair.voxel_idx; }
};

struct grid_t {
	std::vector<int> point_id;
	float min_z;
	float max_z;
	float delta_z;
	float min_z_x; //X of Lowest Point in the Voxel;
	float min_z_y; //Y of Lowest Point in the Voxel;
	float min_z_outlier_thre;
	float neighbor_min_z;
	int pts_count;
	int reliable_neighbor_grid_num;
	float mean_z;
	float dist2station;

	grid_t() {
		min_z = min_z_x = min_z_y = neighbor_min_z = mean_z = 0.f;
		pts_count = 0;
		reliable_neighbor_grid_num = 0;
		delta_z = 0.0;
		dist2station = 0.001;
		min_z_outlier_thre = -FLT_MAX;
	}
};

struct simplified_voxel_t {
	std::vector<int> point_id;
	float max_curvature;
	int max_curvature_point_id;
	bool has_keypoint;
	simplified_voxel_t() {
		has_keypoint = false;
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

	CloudBlock();

	CloudBlock(const CloudBlock &in_block, bool clone_feature = false, bool clone_raw = false);

	void init();

	void init_tree();

	void free_raw_cloud();

	void free_tree();

	void free_all();

	void clone_metadata(const CloudBlock &in_cblock);

	void append_feature(const CloudBlock &in_cblock, bool append_down, std::string used_feature_type);

	void merge_feature_points(pcTPtr &pc_out, bool merge_down, bool with_out_ground = false);

	void transform_feature(const Eigen::Matrix4d &trans_mat, bool transform_down = true, bool transform_undown = true);

	void clone_cloud(pcTPtr &pc_out, bool get_pc_done);

	void clone_feature(pcTPtr &pc_ground_out,
					   pcTPtr &pc_pillar_out,
					   pcTPtr &pc_beam_out,
					   pcTPtr &pc_facade_out,
					   pcTPtr &pc_roof_out,
					   pcTPtr &pc_vertex_out, bool get_feature_down);

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
void get_cloud_bbx(const pcl::PointCloud<MullsPoint>::Ptr &cloud, Bounds &bound);

void get_cloud_bbx_cpt(const pcl::PointCloud<MullsPoint>::Ptr &cloud, Bounds &bound, CenterPoint &cp);

//Get Center of a Point Cloud
void get_cloud_cpt(const typename pcl::PointCloud<MullsPoint>::Ptr &cloud, CenterPoint &cp);

void get_intersection_bbx(Bounds &bbx_1, Bounds &bbx_2, Bounds &bbx_intersection, float bbx_boundary_pad = 2.0);

void merge_bbx(std::vector<Bounds> &bbxs, Bounds &bbx_merged);

} // namespace mulls

#endif //_INCLUDE_MULLS_UTIL_