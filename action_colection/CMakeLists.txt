cmake_minimum_required (VERSION 2.8)

# project name
project (body_points)
# using C++11 
set(CMAKE_CXX_FLAGS "${CAMKE_CXX_FLAGS} -std=c++11 -pthread")

# if u have OpenCV version more than one, set the build path which one u want to use
# find opencv
find_package(OpenCV REQUIRED)
# print message of opencv
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "libraries: ${OpenCV_LIBS}")

aux_source_directory(./src/ DIR_SRCS)

add_executable(body_points  ${DIR_SRCS})

include_directories ( 
    ~/openpose-master/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/local/include
    /usr/include 
${OpenCV_INCLUDE_DIRS})

target_link_libraries(body_points
${OpenCV_LIBS}
/usr/lib
/usr/local/lib
/usr/local/lib/libopenpose.so
~/caffe-master/build/lib/libcaffe.so
/usr/lib/x86_64-linux-gnu/libgflags.so.2
    )
