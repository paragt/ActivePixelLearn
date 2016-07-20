#ifndef _PIXEL_DETECT_WSHED
#define _PIXEL_DETECT_WSHED

#include "pixel_detector.h"
// #define WITH_BOOST_GRAPH
#include "vigra_watershed_uint.h"


class PixelDetectWshed{

    float* feature_data;
    int* groundtruth_data ;
    int depth, width, height, nfeat, nclass, ntrees;
    PixelDetector *pd;
    bool multiclass;
  
public:
    PixelDetectWshed(int pdepth, int pheight, int pwidth, int pnfeat, float* pfeature_data, int pnclass, int pntrees){
	depth = pdepth; 
	height = pheight;
	width = pwidth;
	feature_data = pfeature_data;
	nclass = pnclass;
	multiclass = true;
	nfeat=pnfeat;
	ntrees=pntrees;
      
	pd = new PixelDetector(depth, height, width, nfeat, feature_data, NULL, nclass, ntrees);
	pd->read_data();
    }
    ~PixelDetectWshed(){
    }
    void set_classifier(PixelDetector* ppd){
	pd->set_classifier(ppd);
    }
//     int pp=1;
//     H5Read feature(feature_filename.c_str(), "stack",true);	
//     feature.readData(&feature_data);
//     if (feature.total_dim()==4){
// 	depth = feature.dim()[0];
// 	height = feature.dim()[1];
// 	width = feature.dim()[2];
// 	nfeat = feature.dim()[3];
// 	
//     }
//     
//     printf("feature read\n");
    
    
   
// //     pd.read_data(debug_features, debug_gt);
//     if (feature_data)  	
// 	delete[] feature_data;
//     if (multiclass) pd.load_classifier_m(clfr_name);
//     else pd.load_classifier(clfr_name);
    void compute_watershed(unsigned int* watershed, unsigned int min_cc_sz, unsigned int min_region_sz){
	std::vector< boost::thread* > threads;
	unsigned int ppvolsz = depth*height*width*nclass;
	float* prediction_data = new float[ppvolsz];
	size_t ncores = 8;
	vector < vector<float> >  tmp_prediction_data(ncores);
	float chunkszf = (float) ceil((float)(depth*height*width*1.0/ncores));
	unsigned int chunksz = (unsigned int) chunkszf;
	for(size_t coreid=0; coreid < ncores; coreid++){
	    size_t stidx = coreid*chunksz;
	    size_t enidx = (coreid+1)*chunksz;
	    if (coreid == (ncores-1))
		enidx = depth*height*width;
	    
	    tmp_prediction_data[coreid].resize((enidx-stidx)*nclass);
    // 	cout<< "prediction array size:" << (enidx-stidx) <<endl;
// 	    pd->predict_multiclass_all(tmp_prediction_data[coreid].data(), stidx, enidx);
	    
	    if (multiclass)
		threads.push_back(new boost::thread(&PixelDetector::predict_multiclass_all, pd,
					    tmp_prediction_data[coreid].data(), stidx, enidx));
	    else  
		threads.push_back(new boost::thread(&PixelDetector::predict_multiclass_all, pd,
					    tmp_prediction_data[coreid].data(),stidx,enidx));
	}
	
    //     unsigned int ppvolsz = depth*height*width*nclass;
    //     float* prediction_data = new float[ppvolsz];
    //     if (multiclass) pd.predict_multiclass_all(prediction_data);
    //     else pd.predict_pairwise_all(prediction_data);
	printf("Sync all threads \n");
	for (size_t ti=0; ti<threads.size(); ti++) 
	  (threads[ti])->join();
	printf("all threads done\n");
	
	for(size_t coreid=0; coreid < ncores; coreid++){
	    size_t stidx = coreid*chunksz;
	    size_t enidx = (coreid+1)*chunksz;
	    if (coreid == (ncores-1))
		enidx = depth*height*width;
	    
	    memcpy(prediction_data + (stidx*nclass), tmp_prediction_data[coreid].data(), (enidx-stidx)*nclass*sizeof(float));
	    tmp_prediction_data[coreid].clear();
	}   
	
    //     double* double_prediction = new double[ppvolsz];
    //     for(size_t i=0; i< ppvolsz; i++)
    // 	double_prediction[i] = (double) (prediction_data[i]);
    //     
    //     hsize_t dims_out2[4];
    //     dims_out2[0]=depth; dims_out2[1]= height; dims_out2[2]= width; dims_out2[3] = nclass;  
    //     H5Write(prediction_filename.c_str(),"stack",4,dims_out2, prediction_data);
    //     delete[] double_prediction;

	tmp_prediction_data.clear(); 
	
	unsigned char *prediction_ch0 = NULL;
	unsigned char *seedvol = NULL;
	VigraWatershed vw(depth, height, width);
	vw.get_volume_ptr(&prediction_ch0, &seedvol);
	
	cout<< "prediction done" <<endl;
	unsigned char seed = (unsigned char)(255*0);
	for (unsigned int i=0; i < depth*height*width; i++){
	    prediction_ch0[i] = (unsigned char) (255*(prediction_data[i*nclass+1]));
	    seedvol[i] = (prediction_ch0[i] < 1? 1:0);

	}
	
	delete prediction_data;
	
	cout<< "seeds done" <<endl;
    //     unsigned int* watershed = new unsigned int[depth*height*width];
	
	unsigned int nregions= vw.run_watershed(prediction_ch0, watershed, min_cc_sz, min_region_sz);
	cout<< "watershed done with " << nregions<< " regions."<<endl;
	
// 	hsize_t dims_out[3];
// 	dims_out[0]=depth; dims_out[1]= height; dims_out[2]= width;   
// 	H5Write("test_watershed.h5","stack",3,dims_out, watershed);
    }    
    
};
#endif
