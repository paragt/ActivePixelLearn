CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
project (SemiSupervised)

# IF(NOT CMAKE_BUILD_TYPE)
#     SET (CMAKE_BUILD_TYPE Release)
# ENDIF(NOT CMAKE_BUILD_TYPE)
# 
# SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -w")
# SET(CMAKE_CXX_FLAGS_DEBUG "-ggdb -w")
# SET(CMAKE_CXX_LINK_FLAGS "-O3")
# SET(CMAKE_DEBUG_POSTFIX "-g")

# include_directories (/groups/scheffer/home/paragt/NeuroProof_sparse/SparseSolve/sparse-install/include)
include_directories (/groups/flyem/proj/cluster/toufiq/SuiteSparse/sparse-install/include)

# link_directories (/groups/scheffer/home/paragt/NeuroProof_sparse/SparseSolve/sparse-install/lib/) 
# set (SUITESPARSEDIR /groups/flyem/proj/cluster/toufiq/SuiteSparse/sparse-install/lib )
set (SOURCES kmeans.cpp weightmatrix_iter.cpp)
add_library (SemiSupervised SHARED ${SOURCES})
