
#include <fstream>
#include <sstream>
#include <cassert>
#include <iostream>
#include <memory>

#include <ctime>
#include <cmath>
#include <cstring>

#include "pixel_detector.h"
#include "Watershed/vigra_watershed_uint.h"

using namespace std;



class triplet{
public:
    unsigned int z;
    unsigned int x;
    unsigned int y;
    
    triplet(unsigned int a, unsigned int b, unsigned int c):z(a), y(b), x(c) {};
};

int main(int argc, char** argv){

    string feature_filename;
    string clfr_name;
    string prediction_filename;
    string watershed_filename;
    
    int argc_itr=0;
    unsigned int min_cc_sz=3;
    unsigned int min_region_sz=300;
    bool multiclass = false;
    size_t nclass = 3;
    
    while(argc_itr<argc){
	if (!(strcmp(argv[argc_itr],"-feature"))){
	    feature_filename = argv[++argc_itr];
	}
	if (!(strcmp(argv[argc_itr],"-classifier"))){
	    clfr_name = argv[++argc_itr];
	}
	if (!(strcmp(argv[argc_itr],"-prediction"))){
	    prediction_filename = argv[++argc_itr];
	}
	if (!(strcmp(argv[argc_itr],"-watershed"))){
	    watershed_filename = argv[++argc_itr];
	}

// 	if (!(strcmp(argv[argc_itr],"-threshold"))){
// 	    threshold = atof(argv[++argc_itr]);
// 	}
// 	if (!(strcmp(argv[argc_itr],"-ndata"))){
// 	    ndata = atoi(argv[++argc_itr]);
// 	}
	if (!(strcmp(argv[argc_itr],"-min_cc_sz"))){
	    min_cc_sz = atoi(argv[++argc_itr]);
	}
	if (!(strcmp(argv[argc_itr],"-min_region_sz"))){
	    min_region_sz = atoi(argv[++argc_itr]);
	}
	if (!(strcmp(argv[argc_itr],"-nclass"))){
	    nclass = atoi(argv[++argc_itr]);
	}
// 	if (!(strcmp(argv[argc_itr],"-nomito"))){
// 	    merge_mito = false;
// 	}
// 	if (!(strcmp(argv[argc_itr],"-mito_chull"))){
// 	    merge_mito_by_chull = true;
// 	}
// 	if (!(strcmp(argv[argc_itr],"-read_off"))){
// 	    read_off_wts = true;
// 	}
	if (!(strcmp(argv[argc_itr],"-multiclass"))){
	    multiclass = true;
	}
        ++argc_itr;
    } 	
    
    
    float* feature_data = NULL;
    int* groundtruth_data = NULL;
    int depth, width, height, nfeat;
  

    int pp=1;
    H5Read feature(feature_filename.c_str(), "stack",true);	
    feature.readData(&feature_data);
    if (feature.total_dim()==4){
	depth = feature.dim()[0];
	height = feature.dim()[1];
	width = feature.dim()[2];
	nfeat = feature.dim()[3];
	
    }
    
    printf("feature read\n");
    
    
   
    PixelDetector pd(depth, height, width, nfeat, feature_data, groundtruth_data, nclass,1);
//     pd.read_data(debug_features, debug_gt);
    pd.read_data();
    if (feature_data)  	
	delete[] feature_data;
    if (multiclass) pd.load_classifier_m(clfr_name);
    else pd.load_classifier(clfr_name);
    
    std::vector< boost::thread* > threads;
    unsigned int ppvolsz = depth*height*width*nclass;
    float* prediction_data = new float[ppvolsz];
    size_t ncores = 32;
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
	
	if (multiclass)
	    threads.push_back(new boost::thread(&PixelDetector::predict_multiclass_all, &pd,
					 tmp_prediction_data[coreid].data(), stidx, enidx));
	else  
	    threads.push_back(new boost::thread(&PixelDetector::predict_multiclass_all, &pd,
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
    }   
    
    double* double_prediction = new double[ppvolsz];
    for(size_t i=0; i< ppvolsz; i++)
	double_prediction[i] = (double) (prediction_data[i]);
    
    hsize_t dims_out2[4];
    dims_out2[0]=depth; dims_out2[1]= height; dims_out2[2]= width; dims_out2[3] = nclass;  
    H5Write(prediction_filename.c_str(),"stack",4,dims_out2, prediction_data);
    delete[] double_prediction;
    /* debug
    
    hsize_t dims_out[3];
    dims_out[0]=depth; dims_out[1]= height; dims_out[2]= width;  
    H5Write("check_feat.h5","stack",3,dims_out, debug_features);
    H5Write("check_gt.h5","stack",3,dims_out, debug_gt);
    
    //**/
    
    
    
    
    
    
//     unsigned int ppvolsz = depth*height*width*nclass;
    
    
    
//     float* prediction_data = new float[ppvolsz];
    
// // // // // // // // // // //     pd.predict(prediction_data);
// // // // // // // // // // //     
// // // // // // // // // // // 
    unsigned char *prediction_ch0 = NULL;
    unsigned char *seedvol = NULL;
    VigraWatershed vw(width, height, depth);
    vw.get_volume_ptr(&prediction_ch0, &seedvol);
    
// // // // // // // // // // //     if((!(prediction_ch0)) || (!seedvol)){
// // // // // // // // // // // 	cout<< "pointers not initialized";
// // // // // // // // // // // 	return 0;
// // // // // // // // // // //     }

    cout<< "prediction done" <<endl;
    
// // // //     unsigned char seed = (unsigned char)(255*0);
// // // //     for (unsigned int i=0; i < depth*height*width; i++){
// // // // 	prediction_ch0[i] = (unsigned char) (255*(prediction_data[i*nclass+1]));
// // // // 	seedvol[i] = ( prediction_data[i*nclass+1]  == 0? 1:0);
// // // //     }
    
    unsigned char seed = (unsigned char)(255*0);
    for (size_t dp=0; dp < depth; dp++){
	for(size_t ht=0; ht<height; ht++){
	    for(size_t wd=0; wd<width; wd++){
		size_t source_id = (dp*(height*width*nclass)) + (ht*width*nclass) + (wd*nclass);
		size_t dest_id = (dp*(height*width)) + (ht*width) + (wd);
		
		prediction_ch0[dest_id] = (unsigned char) (255*(prediction_data[source_id+1]));
// 		seedvol[dest_id] = ( prediction_data[source_id+1]  == 0? 1:0);//more oversegmentation
		seedvol[dest_id] = ( prediction_ch0[dest_id]  == 0? 1:0);
	    }
	}
    }
    
    cout<< "seeds done" <<endl;
    unsigned int* watershed = new unsigned int[depth*height*width];
    
    unsigned int nregions= vw.run_watershed(prediction_ch0, watershed, min_cc_sz, min_region_sz);
    cout<< "watershed done with " << nregions<< " regions."<<endl;
    
    
    hsize_t dims_out[3];
    dims_out[0]=depth; dims_out[1]= height; dims_out[2]= width;   
    H5Write(watershed_filename.c_str(),"stack",3,dims_out, watershed);
//     H5Write("fib_trn_wshed.h5","stack",3,dims_out, double_prediction);
    
    
    
    
    
    
    
    
//     std::time_t start, end;
//     std::time(&start);	
//     std::time(&end);	
//     printf("C: Time to update: %.2f sec\n", (difftime(end,start))*1.0);
// 

//     if (feature_data)  	
// 	delete[] feature_data;
    delete [] watershed;
    
    if (prediction_data)  	
	delete[] prediction_data;
    
    
 
}