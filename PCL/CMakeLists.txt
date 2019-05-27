cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pcd_read)

find_package(PCL 1.9 REQUIRED)

set(CMAKE_CXX_STANDARD 14)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (pcd_read readPCL.cpp)
target_link_libraries (pcd_read ${PCL_LIBRARIES})
