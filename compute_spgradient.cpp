
#include <fstream>
#include <sstream>
#include <cassert>
#include <iostream>
#include <memory>

#include <ctime>
#include <cmath>
#include <cstring>

#include "pixel_detect_wshed.h"
#include "compute_vi.h"
// #include "Watershed/vigra_watershed_uint.h"

#define PAIRWISE

using namespace std;



class triplet{
public:
    unsigned int z;
    unsigned int x;
    unsigned int y;
    
    triplet(unsigned int a, unsigned int b, unsigned int c):z(a), y(b), x(c) {};
};




int main(int argc, char** argv){

    string feature_filename;// = argv[1];//"fib_trn_features2.h5";
    string feature_test_filename;// = argv[1];//"fib_trn_features2.h5";
    string groundtruth_test_filename;// = argv[1];//"fib_trn_features2.h5";
    string groundtruth_filename;// = argv[2];//"fib_trn_groundtruth.h5";
    string mask_filename;// = argv[2];//"fib_trn_groundtruth.h5";
    string ctridx_filename;// = argv[3];
    string clfr_name;// = argv[4];

    
    int argc_itr=0;  
    bool multiclass=true;
    bool read_off_wts=false;
    bool two_dim = false;
    size_t ndata=30000;
    size_t start_with=500;
    size_t query_sz=100;
    int max_iter=20;
    size_t nclass = 2;
    double wt_thd = 2.5;
    int ntrees=100;
    while(argc_itr<argc){
	if (!(strcmp(argv[argc_itr],"-feature"))){
	    feature_filename = argv[++argc_itr];
	}
	if (!(strcmp(argv[argc_itr],"-classifier"))){
	    clfr_name = argv[++argc_itr];
	}
	if (!(strcmp(argv[argc_itr],"-centers"))){
	    ctridx_filename = argv[++argc_itr];
	}
	if (!(strcmp(argv[argc_itr],"-groundtruth"))){
	    groundtruth_filename = argv[++argc_itr];
	}

	if (!(strcmp(argv[argc_itr],"-wt_thd"))){
	    wt_thd = atof(argv[++argc_itr]);
	}
	if (!(strcmp(argv[argc_itr],"-ndata"))){
	    ndata = atoi(argv[++argc_itr]);
	}
	if (!(strcmp(argv[argc_itr],"-max_iter"))){
	    max_iter = atoi(argv[++argc_itr]);
	}
	if (!(strcmp(argv[argc_itr],"-query_sz"))){
	    query_sz = atoi(argv[++argc_itr]);
	}
	if (!(strcmp(argv[argc_itr],"-start_with"))){
	    start_with = atoi(argv[++argc_itr]);
	}
	if (!(strcmp(argv[argc_itr],"-nclass"))){
	    nclass = atoi(argv[++argc_itr]);
	}
	if (!(strcmp(argv[argc_itr],"-ntrees"))){
	    ntrees = atoi(argv[++argc_itr]);
	}
// 	if (!(strcmp(argv[argc_itr],"-nomito"))){
// 	    merge_mito = false;
// 	}
// 	if (!(strcmp(argv[argc_itr],"-mito_chull"))){
// 	    merge_mito_by_chull = true;
// 	}
	if (!(strcmp(argv[argc_itr],"-read_off"))){
	    read_off_wts = true;
	}
	if (!(strcmp(argv[argc_itr],"-2D"))){
	    two_dim = true;
	}
	if (!(strcmp(argv[argc_itr],"-test_feature"))){
	    feature_test_filename = argv[++argc_itr];
	}
	if (!(strcmp(argv[argc_itr],"-test_groundtruth"))){
	    groundtruth_test_filename = argv[++argc_itr];
	}
	if (!(strcmp(argv[argc_itr],"-mask"))){
	    mask_filename = argv[++argc_itr];
	}
        ++argc_itr;
    } 	
    
    
    
    
    
    
    
    
    float* feature_data = NULL;
    int* groundtruth_data = NULL;
    int* mask_data=NULL;
    int depth, width, height, nfeat;
  
    float* feature_tst_data = NULL;
    unsigned int* tst_groundtruth_data = NULL;
    int depth_tst, width_tst, height_tst, nfeat_tst;

    int pp=1;
    H5Read feature(feature_filename.c_str(), "stack",true);	
    feature.readData(&feature_data);
    if (feature.total_dim()==4){
	depth = feature.dim()[0];
	height = feature.dim()[1];
	width = feature.dim()[2];
	nfeat = feature.dim()[3];
	
    }
    if(feature_test_filename.size()>0){
	H5Read feature_tst(feature_test_filename.c_str(), "stack",true);	
	feature_tst.readData(&feature_tst_data);
	if (feature_tst.total_dim()==4){
	    depth_tst = feature_tst.dim()[0];
	    height_tst = feature_tst.dim()[1];
	    width_tst = feature_tst.dim()[2];
	    nfeat_tst = feature_tst.dim()[3];
	    
	}
    }
    
    printf("feature read\n");
    
    
    H5Read groundtruth(groundtruth_filename.c_str(), "stack",true);	
    groundtruth.readData(&groundtruth_data);	
 
    if(groundtruth_test_filename.size()>0){
	H5Read tst_groundtruth(groundtruth_test_filename.c_str(), "stack",true);	
	tst_groundtruth.readData(&tst_groundtruth_data);	
    }
    if(mask_filename.size()>0){
	H5Read mask_label(mask_filename.c_str(), "stack",true);	
	mask_label.readData(&mask_data);	
    }
    
    /* debug
    unsigned int* debug_features = new unsigned int[depth*height*width];
    unsigned int* debug_gt = new unsigned int[depth*height*width];
    //**/
//     for(int i=0;i<10;i++)
//       printf("%u ", all_cidx[i]);
//     printf("\n");

    size_t init_sz_class = start_with;
    size_t initsz = (nclass*init_sz_class);
//     size_t init_sz_class = (unsigned int) (ndata*1.0*0.04/nclass);
//     size_t initsz = (nclass*init_sz_class);
    unsigned int chunksz= query_sz;
    size_t ntrnsz=initsz+ max_iter*chunksz;
    size_t incr=1000;
    
    srand(time(0));
    
    PixelDetector pd(depth, height, width, nfeat, feature_data, groundtruth_data, nclass, ntrees);
    if(mask_data){ 
	printf("setting mask..\n");
	pd.set_mask(mask_data);
    }
    pd.read_data();
    std::vector<unsigned int> offset(3,20);
    if (two_dim){
	offset[0]=0; offset[1]=15; offset[2]=15;// for 2D data
    }
    pd.downsample(ndata, offset);
//     pd.downsample(ctridx_filename, ndata);

    
    
    pd.set_initsz_class(init_sz_class);
    srand (time(NULL));
    
//     if (max_iter>0)
// 	pd.build_wtmat(read_off_wts,feature_filename, wt_thd);
    if (max_iter>0){
	pd.build_wtmat(read_off_wts,feature_filename, wt_thd);
	pd.select_initial_points_biased();
    }
    else 
	pd.select_initial_points2();
    

//     pd.select_initial_points_fixed();
//     pd.select_initial_points_biased();
    
    if (multiclass){
	pd.learn_classifier_multiclass();
	pd.predict_multiclass();  
    }
    else{
	pd.learn_classifier_pairwise();
	pd.predict_pairwise();  
    }
    
    
    string clfr_name1 = clfr_name.substr(0, clfr_name.length()-3);
    char tmp[256];
    sprintf(tmp,"%u.h5",0);
    clfr_name1 +="_";
    clfr_name1 += tmp;
    if (multiclass) pd.save_classifier_m(clfr_name1); 
    else pd.save_classifier(clfr_name1); 
    
    PixelDetectWshed* pdw;

    unsigned int *wshed_prev, *wshed_0, *wshed_next;
    ComputeVI* cvi;
    if(groundtruth_test_filename.size()>0){
      
	pdw = new PixelDetectWshed(depth_tst, height_tst, width_tst, nfeat_tst, feature_tst_data, nclass, ntrees);
	pdw->set_classifier(&pd);
	if(!feature_tst_data)
	    delete feature_tst_data;
	cvi = new ComputeVI(depth_tst, height_tst,width_tst);
	wshed_prev = new unsigned int[depth_tst*height_tst*width_tst];
	wshed_0 = new unsigned int[depth_tst*height_tst*width_tst];
	pdw->compute_watershed(wshed_prev, 3, 100);
	memcpy( wshed_0 , wshed_prev, depth_tst*height_tst*width_tst*sizeof(unsigned int));
	wshed_next = new unsigned int[depth_tst*height_tst*width_tst];

	cvi->compute_vi(wshed_prev, tst_groundtruth_data);
    }
    
    
//     size_t nextra = (unsigned int) ((start_with*nclass+ max_iter*query_sz)*0.1);
//     pd.add_extra_samples(nextra);
//     if (multiclass){
// 	pd.learn_classifier_multiclass();
// 	pd.predict_multiclass();  
//     }
//     else{
// 	pd.learn_classifier_pairwise();
// 	pd.predict_pairwise();  
//     }
    
    
    std::time_t start, end;
    std::time(&start);	
    
    size_t trnset_sz_inc = initsz;
    for(size_t iter=1; iter <= max_iter ; iter++){
	printf("iteration: %u:\n", iter);
	
	pd.propagate_labels_multiclass();
	
	bool no_more_samples = pd.add_new_points(chunksz);
	if (no_more_samples)
	  break;
	
	if (multiclass){
	    pd.learn_classifier_multiclass();
	    pd.predict_multiclass();  
	}
	else{
	    pd.learn_classifier_pairwise();
	    pd.predict_pairwise();  
	}
	if ( groundtruth_test_filename.size()>0 && ((iter%10)==0)){
	    pdw->set_classifier(&pd);
	    pdw->compute_watershed(wshed_next,3, 100);
	    double vi = cvi->compute_vi(wshed_next, wshed_prev);
	    double vi_gt = cvi->compute_vi(wshed_next, tst_groundtruth_data);
	    double vi0 = cvi->compute_vi(wshed_0, wshed_next);
	    memcpy( wshed_prev , wshed_next, depth_tst*height_tst*width_tst*sizeof(unsigned int));
	}

	
// 	if (multiclass) pd.save_classifier_inter_m(clfr_name); 
// 	else pd.save_classifier_inter(clfr_name); 
    }
    
//     size_t nextra = (unsigned int) ((start_with+ max_iter*query_sz)*0.1);
//     pd.add_extra_samples(nextra);
//     if (multiclass){
// 	pd.learn_classifier_multiclass();
// 	pd.predict_multiclass();  
//     }
//     else{
// 	pd.learn_classifier_pairwise();
// 	pd.predict_pairwise();  
//     }
    
    std::time(&end);	
    printf("C: Total execution time: %.2f sec\n", (difftime(end,start))*1.0);

    if (multiclass) pd.save_classifier_m(clfr_name);
    else pd.save_classifier(clfr_name);
    pd.trnset_size_class();
    
    
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
// // // // // // // // // // //     unsigned char *prediction_ch0 = NULL;
// // // // // // // // // // //     unsigned char *seedvol = NULL;
// // // // // // // // // // //     VigraWatershed vw(depth, height, width);
// // // // // // // // // // //     vw.get_volume_ptr(&prediction_ch0, &seedvol);
// // // // // // // // // // //     if((!(prediction_ch0)) || (!seedvol)){
// // // // // // // // // // // 	cout<< "pointers not initialized";
// // // // // // // // // // // 	return 0;
// // // // // // // // // // //     }
// // // // // // // // // // //     cout<< "prediction done" <<endl;
// // // // // // // // // // // //     unsigned int* uchar_prediction = new unsigned int[ppvolsz];
// // // // // // // // // // //     unsigned char seed = (unsigned char)(255*0);
// // // // // // // // // // //     for (unsigned int i=0; i < depth*height*width; i++){
// // // // // // // // // // // 	prediction_ch0[i] = (unsigned char) (255*prediction_data[i*nclass+1]);
// // // // // // // // // // // 	seedvol[i] = (prediction_ch0[i] == seed? 1:0);
// // // // // // // // // // //     }
// // // // // // // // // // //     
// // // // // // // // // // //     unsigned int* watershed = new unsigned int[depth*height*depth];
// // // // // // // // // // //     
// // // // // // // // // // //     vw.run_watershed(prediction_ch0, watershed, 5);
// // // // // // // // // // //     cout<< "watershed done" <<endl;
// // // // // // // // // // //     
// // // // // // // // // // //     hsize_t dims_out[3];
// // // // // // // // // // //     dims_out[0]=depth; dims_out[1]= height; dims_out[2]= width;   
// // // // // // // // // // //     H5Write("fib_trn_wshed.h5","stack",3,dims_out, watershed);
    
    
//     hsize_t dims_out[4];
//     dims_out[0]=depth; dims_out[1]= height; dims_out[2]= width; dims_out[3] = nclass;  
//     H5Write("check_bp.h5","stack",4,dims_out, uchar_prediction);
    
    
    
    
    
    
//     std::time_t start, end;
//     std::time(&start);	
//     std::time(&end);	
//     printf("C: Time to update: %.2f sec\n", (difftime(end,start))*1.0);
// 

    if (feature_data)  	
	delete[] feature_data;

    
    if (groundtruth_data)  	
	delete[] groundtruth_data;
    
 
}