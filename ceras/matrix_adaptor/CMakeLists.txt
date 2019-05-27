cmake_minimum_required( VERSION 2.8 )
project( test)
set(OpenCV_DIR /home/ian/software/test/somedir/installation/OpenCV4.0-master/lib/cmake/opencv4)
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_STANDARD 14 )
# set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 添加cmake模块以使用ceres库
# list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


# 寻找Ceres库并添加它的头文件
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )

add_executable( test main.cpp )
# 与Ceres和OpenCV链接
target_link_libraries( test ${CERES_LIBRARIES} ${OpenCV_LIBS} )
