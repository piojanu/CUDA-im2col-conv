cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
set(PROJECT_NAME "CUDA-im2col-conv")

# Find src files and include dir
file(GLOB CPU_SOURCE_FILES "${CMAKE_SOURCE_DIR}/src/*.c")
file(GLOB GPU_SOURCE_FILES "${CMAKE_SOURCE_DIR}/src/*.cu")
set(INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)

# Create lib with CUDA code
find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
  message(FATAL_ERROR "CUDA not found! CMake will exit.")
endif()

cuda_include_directories(${INCLUDE_DIRS})
cuda_add_library(cudalib ${GPU_SOURCE_FILES})
target_include_directories(cudalib PUBLIC ${INCLUDE_DIRS})
