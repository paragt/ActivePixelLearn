CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
project (ActivePixelLearn)

   include_directories (${CMAKE_PREFIX_PATH}/include)

   include_directories (Classifier)
   include_directories (SemiSupervised)
   include_directories (Utilities)
   include_directories (Watershed)

#    link_directories (/groups/flyem/proj/cluster/toufiq/miniconda/envs/env-toufiq/lib) 
#    link_directories (/groups/flyem/proj/cluster/miniconda/envs/flyem/lib) 

   link_directories (${CMAKE_PREFIX_PATH}/lib)
 
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
