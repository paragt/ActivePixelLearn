#include "opencvRFclassifier.h"
#include "assert.h"
// #include <time.h>
#include <ctime>
#include <stdio.h>

OpencvRFclassifier::OpencvRFclassifier(const char* rf_filename){

    _rf=NULL;	
    load_classifier(rf_filename);
     	
}

void  OpencvRFclassifier::load_classifier(const char* rf_filename){
    if(!_rf)
        _rf = new CvRTrees;
    _rf->load(rf_filename);	
    printf("RF loaded with %d trees\n",_rf->get_tree_count());

    if (_trees.size()>0)
	_trees.clear();	
    _tree_count =  _rf->get_tree_count();
    for(int i = 0; i < _tree_count; i++){
	CvForestTree* treep = _rf->get_tree(i);
	_trees.push_back(treep); 	
    }
    _tree_weights.resize(_tree_count, 1.0/_tree_count); 		
     
}

float OpencvRFclassifier::predict(std::vector<float>& pfeatures){
    if(!_rf){
        return(EdgeClassifier::predict(pfeatures));
    }
	//assert(_nfeatures == features.size());

    int nfeatures = pfeatures.size();
    CvMat* features = cvCreateMat(1, nfeatures, CV_32F);	
    float* datap = features->data.fl; 	
    for(int i=0;i< nfeatures;i++)
	datap[i] = pfeatures[i];


    int ntrees = _rf->get_tree_count();
    		
    float prob = 0;
    for(int i=0; i < ntrees; i++){
	int class_idx = _trees[i]->predict(features)->class_idx;
	float resp = (class_idx > 0) ? 1 : 0 ; 	
	
        resp *= _tree_weights[i];

	prob += resp;
    }


    cvReleaseMat( &features );

    return  prob;
			    	
}	


void OpencvRFclassifier::learn(std::vector< std::vector<float> >& pfeatures, std::vector<int>& plabels){

     if (_rf){
	delete _rf;
	_trees.clear();	
	_tree_weights.clear();
     }
     _rf = new CvRTrees;
 	

     int rows = pfeatures.size();
     int cols = pfeatures[0].size();	 	
     
     printf("Number of samples and dimensions: %d, %d\n",rows, cols);
     if ((rows<1)||(cols<1)){
	return;
     }

//      clock_t start = clock();    
     std::time_t start, end;
     std::time(&start);	


     CvMat *features = cvCreateMat(rows, cols, CV_32F);	
     CvMat *labels = cvCreateMat(rows, 1 , CV_32F);	
     float* datap = features->data.fl;
     float* labelp = labels->data.fl;	 	


     int numzeros=0; 	
     for(int i=0; i < rows; i++){
	 labelp[i] = plabels[i];
	 numzeros += ( labelp[i] == -1? 1 : 0 );
	 for(int j=0; j < cols ; j++){
	     datap[i*cols+j]  = (float)pfeatures[i][j];	
	 }
     }
     printf("Number of merge: %d\n",numzeros);



     // 1. create type mask
     CvMat* var_type = cvCreateMat( features->cols + 1, 1, CV_8U );
     cvSet( var_type, cvScalarAll(CV_VAR_NUMERICAL) );
     cvSetReal1D( var_type, features->cols, CV_VAR_CATEGORICAL );

     // define the parameters for training the random forest (trees)

     float priors[] = {1,1};  // weights of each classification for classes
     // (all equal as equal samples of each digit)

     CvRTParams params = CvRTParams( _max_depth, // max depth :the depth of the tree
                                     10, // min sample count:  minimum samples required at a leaf node for it to be split
                                     0, // regression accuracy: N/A here
                                     false, // compute surrogate split, no missing data
                                     15, // max number of categories (use sub-optimal algorithm for larger numbers)
                                     priors, // the array of prior for each class
                                     false,  // calculate variable importance
                                     5,       // number of variables randomly selected at node and used to find the best split(s).
                                     _tree_count,	 // max number of trees in the forest
                                     0.001f, // forest accuracy
                                     CV_TERMCRIT_ITER //|	CV_TERMCRIT_EPS // termination cirteria
                                      );
	

     // 3. train classifier
     _rf->train( features, CV_ROW_SAMPLE, labels, 0, 0, var_type, 0, params);
         //CvRTParams(10,10,0,false,15,0,true,4,100,0.01f,CV_TERMCRIT_ITER));

     float correct = 0;
     for(int  i = 0; i < features->rows ; i++ ){
         float r;
         CvMat sample;
         cvGetRow( features, &sample, i );
	
         r = _rf->predict_prob( &sample );
	 r = (r>0.5)? 1 :-1;
         r = fabs((float)r - labels->data.fl[i]) <= FLT_EPSILON ? 1 : 0;


	 correct += r;
      }
      	

    std::time(&end);
    printf("Time required to learn RF: %.2f sec\n", (difftime(end,start))*1.0);
//     printf("Time required to learn RF: %.2f sec\n", ((float)clock() - start) / CLOCKS_PER_SEC);
    printf("with training set accuracy :%.3f\n", correct/features->rows*100.);

    _tree_count = _rf->get_tree_count();	
    for(int i = 0; i < _tree_count; i++){
	CvForestTree* treep = _rf->get_tree(i);
	_trees.push_back(treep); 	
    }
    //int ntrees = _rf->get_tree_count();	
    _tree_weights.resize(_tree_count, 1.0/_tree_count); 		

     cvReleaseMat( &features );
     cvReleaseMat( &labels );	
     cvReleaseMat( &var_type );	
}

void OpencvRFclassifier::save_classifier(const char* rf_filename){
    _rf->save(rf_filename);	
}

void OpencvRFclassifier::set_tree_weights(vector<float>& pwts){

    assert(pwts.size() == _rf->get_tree_count());	
    float sumwt=0;	  	
    for(int i=0; i< pwts.size(); i++){
	_tree_weights[i] *= pwts[i];	
	sumwt += _tree_weights[i];
    }
    for(int i=0; i< pwts.size(); i++)
	_tree_weights[i] /= sumwt;	
}



void OpencvRFclassifier::get_tree_responses(vector<float>& pfeatures, vector<float>& responses){
    int nfeatures = pfeatures.size();
    CvMat* features = cvCreateMat(1, nfeatures, CV_32F);	
    float* datap = features->data.fl; 	
    for(int i=0;i< nfeatures;i++)
	datap[i] = pfeatures[i];

    int ntrees = _rf->get_tree_count();
    responses.clear();
    responses.resize(ntrees); 	 	
    		
    float prob = 0;
    for(int i=0; i < ntrees; i++){
	float resp = _trees[i]->predict(features)->value;
	resp = (resp > 0) ? 1 : 0 ; 	
	responses[i] = resp*_tree_weights[i];
    }	

}

void OpencvRFclassifier::reduce_trees(){

    for(int i=0; i < _tree_weights.size(); i++)
	if (_tree_weights[i]<0.001)
	    _tree_weights[i] = 0.0;	

}
