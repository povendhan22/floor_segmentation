#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <limits>
#include <cmath>

#include <ros/ros.h>


#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/conditional_removal.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/don.h>
#include <pcl/features/fpfh_omp.h>

#include <pcl/kdtree/kdtree.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include <pcl/common/common.h>

#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>

#include <pcl/segmentation/extract_clusters.h>

#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayLayout.h>
#include <std_msgs/MultiArrayDimension.h>

#include <tf/tf.h>
#include <yaml-cpp/yaml.h>
#include "cluster.h"


#define __APP_NAME__ "euclidean_clustering"



ros::Publisher _pub_ground_cloud;
ros::Publisher _pub_points_lanes_cloud;

std_msgs::Header _velodyne_header;

std::string _output_frame;

static bool _velodyne_transform_available;
static bool _downsample_cloud;
static bool _pose_estimation;
static double _leaf_size;
static int _cluster_size_min;
static int _cluster_size_max;
static const double _initial_quat_w = 1.0;

static bool _remove_ground;  // only ground

static bool _using_sensor_cloud;
static bool _use_diffnormals;

static double _clip_min_height;
static double _clip_max_height;

static bool _keep_lanes;
static double _keep_lane_left_distance;
static double _keep_lane_right_distance;

static double _max_boundingbox_side;
static double _remove_points_upto;
static double _cluster_merge_threshold;
static double _clustering_distance;

std::vector<std::vector<geometry_msgs::Point>> _way_area_points;

pcl::PointCloud<pcl::PointXYZ> _sensor_cloud;

static bool _use_multiple_thres;
std::vector<double> _clustering_distances;
std::vector<double> _clustering_ranges;

tf::StampedTransform *_transform;
tf::StampedTransform *_velodyne_output_transform;
tf::TransformListener *_transform_listener;
tf::TransformListener *_vectormap_transform_listener;


tf::StampedTransform findTransform(const std::string &in_target_frame, const std::string &in_source_frame)
{
  tf::StampedTransform transform;

  try
  {
    // What time should we use?
    _vectormap_transform_listener->lookupTransform(in_target_frame, in_source_frame, ros::Time(0), transform);
  }
  catch (tf::TransformException ex)
  {
    ROS_ERROR("%s", ex.what());
    return transform;
  }

  return transform;
}

geometry_msgs::Point transformPoint(const geometry_msgs::Point& point, const tf::Transform& tf)
{
  tf::Point tf_point;
  tf::pointMsgToTF(point, tf_point);

  tf_point = tf * tf_point;

  geometry_msgs::Point ros_point;
  tf::pointTFToMsg(tf_point, ros_point);

  return ros_point;
}


void publishCloud(const ros::Publisher *in_publisher, const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_to_publish_ptr)
{
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
  

  // printf("Publishing check");
  // printf ("Cloud: width = %d, height = %d\n", (in_cloud_to_publish_ptr)->width, (in_cloud_to_publish_ptr)->height);
  // BOOST_FOREACH (const pcl::PointXYZ& pt, (in_cloud_to_publish_ptr)->points)
  //   printf ("\t(%f, %f, %f)\n", pt.x, pt.y, pt.z);
  
  cloud_msg.header = _velodyne_header;
  in_publisher->publish(cloud_msg);
}

void publishColorCloud(const ros::Publisher *in_publisher,
                       const pcl::PointCloud<pcl::PointXYZRGB>::Ptr in_cloud_to_publish_ptr)
{
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
  cloud_msg.header = _velodyne_header;
  in_publisher->publish(cloud_msg);
}

void keepLanePoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr, float in_left_lane_threshold = 1.5,
                    float in_right_lane_threshold = 1.5)
{
  pcl::PointIndices::Ptr far_indices(new pcl::PointIndices);
  for (unsigned int i = 0; i < in_cloud_ptr->points.size(); i++)
  {
    pcl::PointXYZ current_point;
    current_point.x = in_cloud_ptr->points[i].x;
    current_point.y = in_cloud_ptr->points[i].y;
    current_point.z = in_cloud_ptr->points[i].z;

    if (current_point.y > (in_left_lane_threshold) || current_point.y < -1.0 * in_right_lane_threshold)
    {
      far_indices->indices.push_back(i);
    }
  }
  out_cloud_ptr->points.clear();
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(in_cloud_ptr);
  extract.setIndices(far_indices);
  extract.setNegative(true);  // true removes the indices, false leaves only the indices
  extract.filter(*out_cloud_ptr);
}

void removeFloor(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr out_nofloor_cloud_ptr,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr out_onlyfloor_cloud_ptr, float in_max_height = 0.2,
                 float in_floor_max_angle = 0.1)
{

  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(100);
  seg.setAxis(Eigen::Vector3f(0, 0, 1));
  seg.setEpsAngle(in_floor_max_angle);

  seg.setDistanceThreshold(in_max_height);  // floor distance
  seg.setOptimizeCoefficients(true);
  seg.setInputCloud(in_cloud_ptr);
  seg.segment(*inliers, *coefficients);
  if (inliers->indices.size() == 0)
  {
    std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
  }

  // REMOVE THE FLOOR FROM THE CLOUD
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(in_cloud_ptr);
  extract.setIndices(inliers);
  extract.setNegative(true);  // true removes the indices, false leaves only the indices
  extract.filter(*out_nofloor_cloud_ptr);

  // EXTRACT THE FLOOR FROM THE CLOUD
  extract.setNegative(false);  // true removes the indices, false leaves only the indices
  extract.filter(*out_onlyfloor_cloud_ptr);
}

void downsampleCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr, float in_leaf_size = 0.2)
{
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(in_cloud_ptr);
  sor.setLeafSize((float) in_leaf_size, (float) in_leaf_size, (float) in_leaf_size);
  sor.filter(*out_cloud_ptr);
}

void clipCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr,
               pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr, float in_min_height = -1.3, float in_max_height = 0.5)
{
  out_cloud_ptr->points.clear();
  for (unsigned int i = 0; i < in_cloud_ptr->points.size(); i++)
  {
    if (in_cloud_ptr->points[i].z >= in_min_height && in_cloud_ptr->points[i].z <= in_max_height)
    {
      out_cloud_ptr->points.push_back(in_cloud_ptr->points[i]);
    }
  }
}

void removePointsUpTo(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr,
                      pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr, const double in_distance)
{
  out_cloud_ptr->points.clear();
  for (unsigned int i = 0; i < in_cloud_ptr->points.size(); i++)
  {
    float origin_distance = sqrt(pow(in_cloud_ptr->points[i].x, 2) + pow(in_cloud_ptr->points[i].y, 2));
    if (origin_distance > in_distance)
    {
      out_cloud_ptr->points.push_back(in_cloud_ptr->points[i]);
    }
  }
}

void velodyne_callback(const sensor_msgs::PointCloud2ConstPtr& in_sensor_cloud)
{
  
  if (!_using_sensor_cloud)
  {
    _using_sensor_cloud = true;

    pcl::PointCloud<pcl::PointXYZ>::Ptr current_sensor_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr removed_points_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr inlanes_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr nofloor_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr onlyfloor_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr diffnormals_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr clipped_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
  
    pcl::fromROSMsg(*in_sensor_cloud, *current_sensor_cloud_ptr);

    _velodyne_header = in_sensor_cloud->header;

    if (_remove_points_upto > 0.0)
    {
      removePointsUpTo(current_sensor_cloud_ptr, removed_points_cloud_ptr, _remove_points_upto);
    }
    else
    {
      removed_points_cloud_ptr = current_sensor_cloud_ptr;
    }

    if (_downsample_cloud)
      downsampleCloud(removed_points_cloud_ptr, downsampled_cloud_ptr, _leaf_size);
    else
      downsampled_cloud_ptr = removed_points_cloud_ptr;

    clipCloud(downsampled_cloud_ptr, clipped_cloud_ptr, _clip_min_height, _clip_max_height);
   
    if (_keep_lanes)
      keepLanePoints(clipped_cloud_ptr, inlanes_cloud_ptr, _keep_lane_left_distance, _keep_lane_right_distance);
    else
      inlanes_cloud_ptr = clipped_cloud_ptr;

    if (_remove_ground)
    {
      removeFloor(inlanes_cloud_ptr, nofloor_cloud_ptr, onlyfloor_cloud_ptr);
      
      publishCloud(&_pub_ground_cloud, onlyfloor_cloud_ptr);
      publishCloud(&_pub_points_lanes_cloud, nofloor_cloud_ptr);
    }
    else
    {
      nofloor_cloud_ptr = inlanes_cloud_ptr;
    }
    _using_sensor_cloud = false;
  }
  
}


int main(int argc, char **argv)
{

  // Initialize ROS
  ros::init(argc, argv, "euclidean_cluster");

  ros::NodeHandle h;
  ros::NodeHandle private_nh("~");

  tf::StampedTransform transform;
  tf::TransformListener listener;
  tf::TransformListener vectormap_tf_listener;

  _vectormap_transform_listener = &vectormap_tf_listener;
  _transform = &transform;
  _transform_listener = &listener;

  _pub_ground_cloud = h.advertise<sensor_msgs::PointCloud2>("/points_ground", 1);
 
  _pub_points_lanes_cloud = h.advertise<sensor_msgs::PointCloud2>("/points_lanes", 1);

  std::string points_topic, gridmap_topic;

  _using_sensor_cloud = false;

  _use_diffnormals = false;
  if (private_nh.getParam("use_diffnormals", _use_diffnormals))
  {
    if (_use_diffnormals)
      ROS_INFO("Euclidean Clustering: Applying difference of normals on clustering pipeline");
    else
      ROS_INFO("Euclidean Clustering: Difference of Normals will not be used.");
  }

  /* Initialize tuning parameter */
  private_nh.param("downsample_cloud", _downsample_cloud, true);
  ROS_INFO("[%s] downsample_cloud: %d", __APP_NAME__, _downsample_cloud);
  private_nh.param("remove_ground", _remove_ground, true);
  ROS_INFO("[%s] remove_ground: %d", __APP_NAME__, _remove_ground);
  private_nh.param("leaf_size", _leaf_size, 0.1);
  ROS_INFO("[%s] leaf_size: %f", __APP_NAME__, _leaf_size);
  private_nh.param("cluster_size_min", _cluster_size_min, 20);
  ROS_INFO("[%s] cluster_size_min %d", __APP_NAME__, _cluster_size_min);
  private_nh.param("cluster_size_max", _cluster_size_max, 100000);
  ROS_INFO("[%s] cluster_size_max: %d", __APP_NAME__, _cluster_size_max);
  private_nh.param("pose_estimation", _pose_estimation, false);
  ROS_INFO("[%s] pose_estimation: %d", __APP_NAME__, _pose_estimation);
  private_nh.param("clip_min_height", _clip_min_height, -1.3);
  ROS_INFO("[%s] clip_min_height: %f", __APP_NAME__, _clip_min_height);
  private_nh.param("clip_max_height", _clip_max_height, 0.5);
  ROS_INFO("[%s] clip_max_height: %f", __APP_NAME__, _clip_max_height);
  private_nh.param("keep_lanes", _keep_lanes, true);
  ROS_INFO("[%s] keep_lanes: %d", __APP_NAME__, _keep_lanes);
  private_nh.param("keep_lane_left_distance", _keep_lane_left_distance, 5.0);
  ROS_INFO("[%s] keep_lane_left_distance: %f", __APP_NAME__, _keep_lane_left_distance);
  private_nh.param("keep_lane_right_distance", _keep_lane_right_distance, 5.0);
  ROS_INFO("[%s] keep_lane_right_distance: %f", __APP_NAME__, _keep_lane_right_distance);
  private_nh.param("max_boundingbox_side", _max_boundingbox_side, 10.0);
  ROS_INFO("[%s] max_boundingbox_side: %f", __APP_NAME__, _max_boundingbox_side);
  private_nh.param("cluster_merge_threshold", _cluster_merge_threshold, 1.5);
  ROS_INFO("[%s] cluster_merge_threshold: %f", __APP_NAME__, _cluster_merge_threshold);
  private_nh.param<std::string>("output_frame", _output_frame, "velodyne");
  ROS_INFO("[%s] output_frame: %s", __APP_NAME__, _output_frame.c_str());

  private_nh.param("remove_points_upto", _remove_points_upto, 0.0);
  ROS_INFO("[%s] remove_points_upto: %f", __APP_NAME__, _remove_points_upto);

  private_nh.param("clustering_distance", _clustering_distance, 0.75);
  ROS_INFO("[%s] clustering_distance: %f", __APP_NAME__, _clustering_distance);

  
  private_nh.param("use_multiple_thres", _use_multiple_thres, false);
  ROS_INFO("[%s] use_multiple_thres: %d", __APP_NAME__, _use_multiple_thres);

  std::string str_distances;
  std::string str_ranges;
  private_nh.param("clustering_distances", str_distances, std::string("[0.5,1.1,1.6,2.1,2.6]"));
  ROS_INFO("[%s] clustering_distances: %s", __APP_NAME__, str_distances.c_str());
  private_nh.param("clustering_ranges", str_ranges, std::string("[15,30,45,60]"));
    ROS_INFO("[%s] clustering_ranges: %s", __APP_NAME__, str_ranges.c_str());

  _velodyne_transform_available = false;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = h.subscribe("/velodyne_points", 1, velodyne_callback);

  // Spin
  ros::spin();
}

