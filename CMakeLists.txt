cmake_minimum_required(VERSION 3.1)
# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# SET(OpenCV_DIR /home/ian/software/test/somedir/installation/OpenCV4.0-master/lib/cmake/opencv4)

include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package( OpenCV REQUIRED )

#include_directories("/home/ian/software/test/somedir/installation/OpenCV4.0-master/include/opencv4
#")

# add_library("/home/ian/software/test/somedir/installation/OpenCV4.0-master/lib/*.so")


set(SRC_LIST mat_the_basic_image_container.cpp)
# set(SRC_LIST show.cpp)


add_executable(matrix ${SRC_LIST})

target_link_libraries(matrix ${OpenCV_LIBS})
