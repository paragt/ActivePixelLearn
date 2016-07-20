# -*- coding: utf-8 -*-

#!/usr/bin/env python

#from ray import imio, agglo, morpho, classify, features, evaluate

#from skimage import morphology as skmorph
#from scipy.ndimage import label

import StdFeatures

from numpy import unique
import sys
import h5py
import pdb

if __name__ == "__main__" :
  
    if len(sys.argv)<2:
	print "Format: python compute_pixel_features.py ..../grayscale_maps/*.png feature_filename.h5"
    data_file = sys.argv[1:]
    feature_file = data_file[-1:]
    data_file = data_file[:-1]
    #feature_file = sys.argv[2]
    
    #pdb.set_trace()
    
    feat_obj=StdFeatures.FeatCompute()
    
    F = feat_obj.compute_ilastik_features(data_file)
    
    #pdb.set_trace()
    f= h5py.File(feature_file[0],'w')
    f.create_dataset('stack', data=F)
    f.close()
    
    #prediction_file=sys.argv[1]
    #watershed_file= sys.argv[2]

    #seed_cc_threshold = 5

    #p = imio.read_image_stack(prediction_file, group= 'volume/predictions', single_channel=False)
    #p0 = p[..., 0]
    #print "Performing watershed"
    #seeds = label(p0==0)[0]
    #if seed_cc_threshold > 0:
       #seeds = morpho.remove_small_connected_components(seeds, seed_cc_threshold)
    #ws = skmorph.watershed(p0, seeds)
    #ws_int, dummy1, dummy2 = evaluate.relabel_from_one(ws)
    #ws_int1 =  ws_int.astype('int32')
    
    #imio.write_h5_stack(ws_int1,watershed_file); 	
    #print "Imported first stack with "+ str(unique(ws_int1).size) + " regions"
