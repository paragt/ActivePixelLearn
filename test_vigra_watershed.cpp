
#include <fstream>
#include <sstream>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>

#include <ctime>
#include <cmath>
#include <cstring>
#include "Utilities/h5read.h"
#include "Utilities/h5write.h"

#include "Watershed/vigra_watershed_uint.h"

using namespace std;

int main(int argc, char** argv){

    string prediction_filename = argv[1];//"/groups/scheffer/home/paragt/Neuroproof_lean/NeuroProof_toufiq/tmpdata/anirban/pixel-prob/STACKED_prediction.h5";
    string volume_name = argv[2];//"volume/predictions";
    string output_volname = argv[3]; //"/groups/scheffer/home/paragt/pixel_classify/tst_anirban_wshed.h5";
    int min_cc_sz = atoi(argv[4]);
    int min_region_sz = atoi(argv[5]);

    printf("prediction file: %s\n", prediction_filename.c_str());
    printf("watershed file: %s\n", output_volname.c_str());
    
    float* prediction_data;
    int depth, width, height, nfeat;

    H5Read prediction(prediction_filename.c_str(), volume_name.c_str(),true);	
    prediction.readData(&prediction_data);
    if (prediction.total_dim()==4){
	depth = prediction.dim()[0];
	height = prediction.dim()[1];
	width = prediction.dim()[2];
	nfeat = prediction.dim()[3];
	
    }
    else if (prediction.total_dim()==3){
	depth = prediction.dim()[0];
	height = prediction.dim()[1];
	width = prediction.dim()[2];
	nfeat = 1;
	
    }
    time_t start, end;
    time(&start);	
    
    //unsigned char* prediction_ch0 = new unsigned char[depth*height*depth];
    
    unsigned char *prediction_ch0 = NULL;
    unsigned char *seedvol = NULL;
    VigraWatershed vw(width, height,depth);
    vw.get_volume_ptr(&prediction_ch0, &seedvol);
    
    if((!(prediction_ch0)) || (!seedvol)){
	cout<< "pointers not initialized";
	return 0;
    }
    
    size_t cube_size, plane_size;
    size_t position, position_l, count, i, j, k;		    	
    count = 0;
    
    //*C* debug
    FILE* fp=fopen("prediction_data.txt", "wt");
    
    size_t element_size= nfeat;
    for(i=0; i<depth; i++){
	cube_size = height*width*element_size;	
	for(j=0; j<height; j++){
	    plane_size = width*element_size;
	    for(k=0; k<width; k++){
	        
		position = i*cube_size + j*plane_size + k*element_size ;
		position_l = i*(cube_size/element_size) + j*(plane_size/element_size) + k ;
		
		 
		
		prediction_ch0[position_l] = (unsigned char)(255* prediction_data[position+1]);
		seedvol[position_l] = (prediction_ch0[position_l] == 0? 1 : 0);
		
		//*C* debug
		if (i<1){
// 		  for(size_t cc=0;cc<element_size;cc++)
// 		      fprintf(fp,"%f  ", prediction_data[position+cc]);
// 		  fprintf(fp,"\n");
		    fprintf(fp,"%d\n", prediction_ch0[position_l]);
		}
		//*C* debug
		
	    }		
	}	
    }
    
    
    fclose(fp);
    
    
// //     hsize_t dims_out[3];
// //     dims_out[0]=depth; dims_out[1]= height; dims_out[2]= width;   
// //     H5Write("check_bp.h5","stack",3,dims_out, prediction_ch0);
    
    unsigned int* watershed = new unsigned int[depth*height*depth];
    
    
    unsigned int nregions = vw.run_watershed(prediction_ch0, watershed, (unsigned) min_cc_sz,(unsigned) min_region_sz);
    cout<< "watershed done with " << nregions<< " regions."<<endl;

// // //     ofstream myfile ("tstwshed.bin", ios::binary);
// // //     myfile.write((char*) watershed, depth*height*width*sizeof(unsigned int));
// // //     myfile.close();
	
    hsize_t dims_out[3];
    dims_out[0]=depth; dims_out[1]= height; dims_out[2]= width;   
    H5Write(output_volname.c_str(),"stack",3,dims_out, watershed);
    
    time(&end);	
    printf("Time elapsed: %.2f sec\n", (difftime(end,start))*1.0);
    
    if (prediction_data)  	
	delete[] prediction_data;

    if (watershed)  	
	delete[] watershed;

    return 0;
}