cmake_minimum_required(VERSION 3.13)
project(nh)

set(OpenCV_DIR /home/ian/software/test/somedir/installation/OpenCV4.0-master/lib/cmake/opencv4)

set(CMAKE_CXX_STANDARD 14)

#FIND_PACKAGE( OpenMP REQUIRED)
#if(OPENMP_FOUND)
#    message("OPENMP FOUND")
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

# list(APPEND CMAKE_INCLUDE_PATH /usr/local/cuda-9.2/include)
# list(APPEND CMAKE_LIBRARY_PATH /usr/local/cuda-9.2/lib64)
# set(CUDA_PATH /usr/local/cuda-9.2)
# set(CUDA_DIR /usr/local/cuda-9.2)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-9.2)
# set(CUDA_VERSION 9.2)
# find_package(CUDA REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
find_package( OpenCV REQUIRED )

add_executable(nh nonmaxup.cpp)

target_link_libraries(nh ${OpenCV_LIBS})
