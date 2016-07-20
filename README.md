# ActivePixelLearn

This repo is the implementation of the ICCV 2015 paper: 

Parag, T. et.al. (2015).  Efficient Classifier Training to Minimize False Merges in Electron Microscopy Segmentation. In Proc. of ICCV 2015. (https://www.researchgate.net/publication/287215311_Efficient_Classifier_Training_to_Minimize_False_Merges_in_Electron_Microscopy_Segmentation) 

Although the method was developed with an interactive interface in mind, i.e., where the user would be asked for labels of individual pixels, the codes provided here needs the actual labels to be provided with the pixel features. 

Once a superpixel boundary has been learned using these codes, it can be used for agglomeration using the codes at: 
 https://github.com/paragt/Neuroproof_minimal

and

 https://github.com/paragt/ActiveSpBdryLearn



# Build

Linux: Install miniconda on your workstation. Create and activate the conda environment using the following commands:

  conda create -n my_conda_env -c flyem vigra opencv 

  source activate my_conda_env

Then follow the usual procedure of building:

  mkdir build
 
  cd build

  cmake -DCMAKE_PREFIX_PATH=[CONDA_ENV_PATH]/my_conda_env ..


# Example

Given a set of grayscale images, one first needs to compute the pixel features using the following python script:

python compute_pixel_features.py grayscale_maps/*.png  data/example/pixel_features_o.h5

The features are then used to train the pixel classifier. The pixel classifier can be applied to a new feature set using the following command:

build/predict_pixels -feature pixel_features_o.h5 -classifier pixel_classifier_4class_2.5_600000_10_800_1000_1.0_1.h5  -prediction  pixel_prediction2_4class_600000_10_800_1000_1.0_1.h5 -watershed watershed2_4class_600000_10_800_1000_1.0_1.h5 -multiclass -nclass 4 -min_cc_sz 1 -min_region_sz 300

nclass implies the number of classes the pixel detector is trained for, min_cc_sz is the size of connected components with membrane prob < 0.00001, and min_region_sz is the size threshold on the resulting oversegmentation performed by watershed.

A sample set of images and the learned pixel predictor is saved in https://www.dropbox.com/sh/u5v9hbz9i5s3u7z/AAD4RTew7PCa0kKWNyJp8jVxa?dl=0 