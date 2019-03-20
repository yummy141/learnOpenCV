cmake_minimum_required(VERSION 3.13)
project(untitled)

set(OpenCV_DIR /home/ian/software/test/somedir/installation/OpenCV4.0-master/lib/cmake/opencv4)
set(CMAKE_CXX_STANDARD 14)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()



include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package( OpenCV REQUIRED )


add_executable(untitled main.cpp)

target_link_libraries(untitled ${OpenCV_LIBS})