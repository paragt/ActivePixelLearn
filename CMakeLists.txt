CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
project (goo)

# # include (ExternalProject)
# # 
# # ############################################################################
# # # Check if BUILDEM_DIR has already been assigned.  If not, create a default.
# # set (BUILDEM_DIR "None" CACHE TYPE STRING)
# # 
# # if (${BUILDEM_DIR} STREQUAL "None")
# #     message (FATAL_ERROR "ERROR: FlyEM build directory (for all downloads & builds) should be specified via -D BUILDEM_DIR=<path> on cmake command line.")
# # endif ()
# # 
# # message ("FlyEM downloads and builds will be placed here: ${BUILDEM_DIR}")
# # 
# # ############################################################################
# # 
# # ############################################################################
# # # Download and install buildem, if it isn't already in BUILDEM_DIR.
# # set (BUILDEM_REPO_DIR ${BUILDEM_DIR}/src/buildem)
# # if (NOT EXISTS ${BUILDEM_REPO_DIR}/python.cmake)
# #     message ("Installing buildem repo...")
# #     ExternalProject_Add(buildem
# #         PREFIX              ${BUILDEM_DIR}
# #         GIT_REPOSITORY      https://github.com/janelia-flyem/buildem.git
# #         #GIT_TAG            python3  # Example of tagged branch (see doc)
# #         UPDATE_COMMAND      ""
# #         PATCH_COMMAND       ""
# #         CONFIGURE_COMMAND   "" 
# #         BUILD_COMMAND       ""
# #         BUILD_IN_SOURCE     1
# #         INSTALL_COMMAND     ""
# #     )
# #     message ("\n**********************************************************\n")
# #     message ("\nAfter running make, you must re-run the cmake command once")
# #     message ("buildem has been downloaded!\n")
# #     message ("\n***********************************************************\n")
# # else ()
# #     ############################################################################
# # 
# #     # Use modules from the downloaded buildem
# #     set (CMAKE_MODULE_PATH ${BUILDEM_REPO_DIR})
# #     message("Using cmake modules from ${BUILDEM_REPO_DIR}")
# # 
# #     # Download, compile, test, and install dependencies
# #     # Note the auto-generated APP_DEPENDENCIES variable holds all included targets.
# #     include (boost)
# #     include (vigra)
# #     include (opencv)
# #     include (hdf5)
# # 
# #     add_custom_target (AppDependencies ALL
# #         DEPENDS ${APP_DEPENDENCIES}
# #         COMMENT "Installed all required software using buildem.")
# # 
# #     # Install Foo -- actual build commands should go below
# #     # add_executable(...)

#  ask to use -DCMAKE_PREFIX_PATH=/groups/scheffer/home/paragt/miniconda/envs/tst_my/

#    include_directories (/groups/flyem/proj/cluster/toufiq/miniconda/envs/env-toufiq/include/)
#    include_directories (/groups/flyem/proj/cluster/miniconda/envs/flyem/include/)

   include_directories (Classifier)
   include_directories (SemiSupervised)
   include_directories (Utilities)
   include_directories (Watershed)

#    link_directories (/groups/flyem/proj/cluster/toufiq/miniconda/envs/env-toufiq/lib) 
#    link_directories (/groups/flyem/proj/cluster/miniconda/envs/flyem/lib) 

   add_subdirectory (Utilities)
   add_subdirectory (Classifier)
   add_subdirectory (SemiSupervised)

    set (CMAKE_CXX_FLAGS "-std=c++0x")

   add_executable (compute_spgradient compute_spgradient.cpp pixel_detector.cpp) 
   add_executable (predict_pixels predict_pixels.cpp pixel_detector.cpp) 
   add_executable (test_vigra_watershed test_vigra_watershed.cpp) 


#    add_library(Classifier SHARED Classifier/opencvRFclassifier.cpp Classifier/vigraRFclassifier.cpp)
#    add_library(SemiSupervised SHARED SemiSupervised/kmeans.cpp SemiSupervised/weightmatrix1.cpp)

   target_link_libraries (compute_spgradient Classifier SemiSupervised Utilities vigraimpex opencv_ml opencv_core boost_thread boost_system boost_chrono hdf5 hdf5_hl)  
   target_link_libraries (predict_pixels Classifier SemiSupervised Utilities vigraimpex opencv_ml opencv_core boost_thread boost_system boost_chrono hdf5 hdf5_hl)  
   target_link_libraries (test_vigra_watershed Utilities vigraimpex opencv_ml opencv_core boost_thread boost_system boost_chrono hdf5 hdf5_hl)  


############################################################################
# endif()