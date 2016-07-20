#!/usr/bin/env python
# -*- coding: utf-8 -*-


#execfile('set_paths.py')

import vigra
import numpy
import h5py

import pdb
from PIL import Image


def differenceOfGaussians(data, sigma):
    return vigra.filters.gaussianSmoothing(data, sigma) - vigra.filters.gaussianSmoothing(data, sigma * 0.66)

def hessianOfGaussianEigenvalues(data, sigma):
    return vigra.filters.tensorEigenvalues(vigra.filters.hessianOfGaussian3D(data, sigma))


class FeatCompute:


        #func = differenceOfGaussians





    def __init__(self):	
        self.availableFeats= { 'GGRAD' : vigra.filters.gaussianGradient, \
			   'GaussianGradientMagnitude': vigra.filters.gaussianGradientMagnitude, \
			   'HESSGAUSS': vigra.filters.hessianOfGaussian3D, \
			   'HessianOfGaussianEigenvalues' :  hessianOfGaussianEigenvalues, \
			   'STRUCTENS' : vigra.filters.structureTensor, \
			   'StructureTensorEigenvalues' : vigra.filters.structureTensorEigenvalues, \
			   'GaussianSmoothing' : vigra.filters.gaussianSmoothing, \
			   'LaplacianOfGaussian' : vigra.filters.laplacianOfGaussian, \
			   'DifferenceOfGaussians' : differenceOfGaussians}		


    def applyFeat(self,data,feat_names,sigmas, save_result=False):

	filters=[];
	for ff in feat_names:
	    if ff in self.availableFeats.keys(): 
		filters.append(self.availableFeats[ff])

	#save_result = False
        result=numpy.array([])
        frespannot=[]
  
	#pdb.set_trace()
        for ff in range(len(filters)): 
            #for sgi in range(len(sigmas)):
	    sg = sigmas[ff]
	    print 'vigra compute {0} with sigma {1}'.format(feat_names[ff],sg)
	    filt=filters[ff];
	    
	    if filt == 	vigra.filters.structureTensor or filt == vigra.filters.structureTensorEigenvalues:
		tres = filt(data, sg, sg / 2.0)
	    else:	
		tres = filt(data, sg)
		#tres = filt(data[...,i], sg, window_size=3.5) # stuart berg

	    #if len(tres.shape) < len(data.shape):
	      #tres.shape = tres.shape + (1,)

	     
	    if len(result.shape) < len(data.shape):
		if len(data.shape) < len(tres.shape):
		    result = tres
		else:
		    tres1 = numpy.expand_dims(tres, axis = 3)
		    result = tres1
	    else:
		if  len(tres.shape) == len(data.shape):
		    tres1 = numpy.expand_dims(tres, axis = 3)
		    result = numpy.concatenate((result, tres1),axis = 3)
		elif len(data.shape) < len(tres.shape):
		    for dd in range(tres.shape[3]):
			tres1 = numpy.expand_dims(tres[...,dd], axis = 3)
			
			result = numpy.concatenate((result, tres1),axis = 3)
			
		    
	    #result.append(tres)
    

	    #for j in range(tres.channels):
		#astr= feat_names[ff] + '_sigma'+ str(sg) + '_imchannel'+ str(i)+'_outchannel'+str(j)
		#frespannot.append(astr)	

	    #if save_result:
		#for j in range(tres.channels):
		    #outnamex= 'output_'+ feat_names[ff] + '_sigma'+ str(sg) + '_imchannel'+ \
				#str(i)+'_outchannel'+str(j) +'.jpg'
		    #vigra.impex.writeImage(tres[...,j],outnamex,dtype='',compression='')

	#pdb.set_trace()
        #allfeat = numpy.concatenate(result, axis=-1)	

	return result

    def read_ilp_selection(self, ilpfilename):
	f= h5py.File(ilpfilename,'r')
	scales = numpy.array(f['FeatureSelections']['Scales'])
	filters = numpy.array(f['FeatureSelections']['FeatureIds'])
	selection_matrix = numpy.array(f['FeatureSelections']['SelectionMatrix']);
    
	f.close()
	return filters, scales, selection_matrix
	
    def read_data(self, filename):
	
	#pdb.set_trace()
	if len(filename)==1:
	    filename=filename[0] 
	    extn = filename[-3:]
	    if extn=='.h5':
		f= h5py.File(filename,'r')
		data=numpy.array(f['stack']).astype('float32')
		f.close()
	else: # filename should be ..../grayscale_maps/*.png
	    filelist=filename
	    tmparr = numpy.array(Image.open(filelist[0]))	
	    height = tmparr.shape[0]
	    width = tmparr.shape[1]
	    nfiles = len(filelist)
	    data = numpy.zeros((nfiles, height, width)).astype('float32')
	    for i in range(nfiles):
		tmparr = numpy.array(Image.open(filelist[i]))	
		data[i,:,:] = tmparr
		
	return data
	
    def compute_ilastik_features(self, volname, ilpname=""):
	if ilpname=="":
	    ilpname = '../Segmentation_pipeline/trained_ilps/MyProject_ordishc_8_plus200.ilp'
	
	allfeat_names, all_sigma, selection_matrix = self.read_ilp_selection(ilpname)
	
	feat_names=[]
	sigmas=[]
	for ni in range(len(allfeat_names)):
	    for si in range(len(all_sigma)):
		if selection_matrix[ni][si] == True:
		    feat_names.append(allfeat_names[ni])
		    sigmas.append(all_sigma[si])
    
	data = self.read_data(volname)
	
	#pdb.set_trace()
	features = self.applyFeat(data, feat_names, sigmas)
	return features  
    #sigmas=[]
    #filters=[]
    #annot=[]

    #sigmas.append([0.5,  3.5])
    #filters.append()
    #annot.append('gausgrad')



    #sigmas.append([0.5, 1, 1.5])
    #filters.append(vigra.filters.hessianOfGaussian2D)
    #annot.append('hesgaus2')

