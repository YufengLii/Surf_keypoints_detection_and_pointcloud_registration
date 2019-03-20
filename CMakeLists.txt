cmake_minimum_required(VERSION 2.8)
set( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ${CMAKE_MODULE_PATH} )
project( Surf_keypoints_detection_and_pointcloud_registration )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./bin)

#set( CMAKE_CXX_FLAGS "-std=c++11" )

find_package( OpenCV 3 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


find_package(PCL 1.8 REQUIRED)
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )
add_definitions( -DPCL_NO_PRECOMPILE )
link_directories( ${PCL_LIBRARY_DIRS} )



add_executable( Surf_detection_with_Ransac Surf_detection_with_Ransac.cpp )
target_link_libraries( Surf_detection_with_Ransac ${OpenCV_LIBS} )


add_executable( Surf_rgbd_pointcloud_Registration Surf_rgbd_pointcloud_Registration.cpp )
target_link_libraries( Surf_rgbd_pointcloud_Registration ${OpenCV_LIBS} )
target_link_libraries( Surf_rgbd_pointcloud_Registration ${OpenCV_LIBS} ${PCL_LIBRARIES})
