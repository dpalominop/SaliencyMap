cmake_minimum_required( VERSION 3.5.0 )
project( SaliencyMap )

set( CMAKE_BUILD_TYPE Debug )

set( CMAKE_C_COMPILER /usr/bin/gcc-6 )
set( CMAKE_CXX_COMPILER /usr/bin/g++-6 )

set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_FLAGS_DEBUG "-g -fopenmp -Wno-unused-parameter" )

find_package( CUDA REQUIRED )
find_package( OpenCV REQUIRED )

include_directories( ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} common/inc/ )

cuda_add_library( SaliencyMapLibrary
				  Filter/Filter.cpp
				  Filter/Filter.cu
				  tools/SaliencyMap/SaliencyMap.cpp
				  tools/Filter/Filter.cpp
				  SaliencyMap/SaliencyMap.cpp
				  SaliencyMap/SaliencyMap.cu )
target_link_libraries( SaliencyMapLibrary
					   ${OpenCV_LIBS} )


add_executable( main
				SaliencyMap/kernel.cpp
				SaliencyMap/main.cpp )
target_link_libraries( main
					   SaliencyMapLibrary )