cmake_minimum_required(VERSION 3.4)
set(CMAKE_CXX_STANDARD 14)

message("Current Cmake version is : " ${CMAKE_VERSION})

project(lidar_odom CXX)
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} $ENV{CXXFLAGS} -Wall -Wno-unused-variable -Werror=return-type")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -ggdb3 -DNDEBUG")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules")

########################
# find libs & include  #
########################

set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)

find_package(OpenMP REQUIRED)
include_directories(${OpenMP_FOUND_INCLUDE_DIRS})
message("OpenMP [OK]:" ${OpenMP_FOUND_INCLUDE_DIRS})

#Eigen (involved in pcl, not needed)
find_package(Eigen3 REQUIRED NO_MODULE QUIET)
message("Eigen3 [OK]")

#PCL (neccessary)
#Boost, vtk, flann, eigen are involved in PCL
find_package(PCL REQUIRED QUIET)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
message("PCL [OK]:" ${PCL_INCLUDE_DIRS})

#GFLAG (neccessary)
find_package(GFLAGS REQUIRED QUIET)
include_directories(${GFLAGS_INCLUDE_DIRS})
message("GFLAGS [OK]:" ${GFLAGS_INCLUDE_DIRS})

#GLOG (neccessary)
find_package(GLOG REQUIRED QUIET)
include_directories(${GLOG_INCLUDE_DIRS})
message("GLOG [OK]:" ${GLOG_INCLUDE_DIRS})

#TEASER++
find_package(teaserpp REQUIRED QUIET)
include_directories(${teaserpp_INCLUDE_DIRS})
message("Teaser++ [OK]")
add_definitions(-DTEASER_ON)

# include folder
include_directories(${PROJECT_SOURCE_DIR}/include)

# source folder
set(SRC_LIST
  include/cregistration.hpp
  src/mulls_filter.cpp
  src/mulls_util.cpp
  src/pca.cpp
  )

########################
#      link libs       #
########################
set(DEP_LIBS ${DEP_LIBS} Threads::Threads Boost::thread OpenMP::OpenMP_CXX)
set(DEP_LIBS ${DEP_LIBS} Eigen3::Eigen)
#link pcl lib (neccessary)
set(DEP_LIBS ${DEP_LIBS} ${PCL_LIBRARIES})
set(DEP_LIBS ${DEP_LIBS} ${GFLAGS_LIBRARIES})
set(DEP_LIBS ${DEP_LIBS} ${GLOG_LIBRARIES})
#link teaser ++ (optional)
set(DEP_LIBS ${DEP_LIBS} teaserpp::teaser_registration teaserpp::teaser_io)

########################
#   add executables    #
########################


add_executable(test ${PROJECT_SOURCE_DIR}/test.cpp ${SRC_LIST})
target_link_libraries(test ${DEP_LIBS})

