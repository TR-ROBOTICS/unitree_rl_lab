cmake_minimum_required(VERSION 3.12)
project(g1_29dof_controller)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

# Select onnxruntime by architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ONNXRUNTIME_DIR ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-aarch64-1.22.0)
else()
    set(ONNXRUNTIME_DIR ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0)
endif()

include_directories(
  /usr/include/eigen3
  /opt/unitree_robotics/include/ddscxx # dds (sdk2 installed under /opt here)
  /usr/local/include/iceoryx/v2.0.2 # dds (optional; absent on this host)
  include
  ${ONNXRUNTIME_DIR}/include
  ${PROJECT_SOURCE_DIR}/../../include/
  /opt/unitree_robotics/include # unitree_sdk2 + dds_wrapper headers
)

link_directories(/opt/unitree_robotics/lib)

link_libraries(
  unitree_sdk2 ddsc ddscxx rt pthread # dds
  libboost_program_options.a libyaml-cpp.a fmt
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.22.0
)

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(g1_ctrl main.cpp)