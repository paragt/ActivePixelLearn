#include "vigraRFclassifier.h"
#include "assert.h"
// #include <time.h>
#include <ctime>

VigraRFclassifier::VigraRFclassifier(const char* rf_filename){

    _rf=NULL;	
    load_classifier(rf_filename);
}

void  VigraRFclassifier::load_classifier(const char* rf_filename){
     	HDF5File rf_file(rf_filename,HDF5File::Open); 	
	if(!_rf)
		_rf = new RandomForest<>;
     	rf_import_HDF5(*_rf, rf_file,"rf");
	printf("RF loaded with %d trees, for %d class prob with %d dimensions\n",_rf->tree_count(), _rf->class_count(), _rf->column_count());
	_nfeatures = _rf->column_count();
        _nclass = _rf->class_count();
     
}

float VigraRFclassifier::predict(std::vector<float>& features){
	if(!_rf){
	    printf("no RF learned, returning random prediction\n");
	    return(EdgeClassifier::predict(features));
	}
	assert(_nfeatures == features.size());
        MultiArray<2, float> vfeatures(Shape(1,_nfeatures));
     	MultiArray<2, float> prob(Shape(1, _nclass));
	for(int i=0;i<_nfeatures;i++)
	      vfeatures(0,i)= (float)features[i];


        _rf->predictProbabilities(vfeatures, prob);    

	/*debug*/
	std::vector<float> vp(_nclass);
	for(int i=0;i<_nclass;i++)
	    vp[i] = prob(0,i);	
	/**/

	return (float) prob(0,1);
			    	
}	

void VigraRFclassifier::predict_m(std::vector<float>& features,
				  std::vector<float>& class_prob)
{
	if(!_rf){
	    printf("no RF learned, returning \n");
	}
	assert(_nfeatures == features.size());
        MultiArray<2, float> vfeatures(Shape(1,_nfeatures));
     	MultiArray<2, float> prob(Shape(1, _nclass));
	for(int i=0;i<_nfeatures;i++)
	      vfeatures(0,i)= (float)features[i];

        _rf->predictProbabilities(vfeatures, prob);    

	if (class_prob.size()!=_nclass){
	    class_prob.clear(); 
	    class_prob.resize(_nclass);
	}
	    
	for(int i=0;i<_nclass;i++)
	   class_prob[i] = prob(0,i);
	
	
}	

void VigraRFclassifier::learn(std::vector< std::vector<float> >& pfeatures, std::vector<int>& plabels){

     if (_rf)
	delete _rf;	

     int rows = pfeatures.size();
     int cols = pfeatures[0].size();	 	
     
     printf("Number of samples and dimensions: %d, %d\n",rows, cols);
     if ((rows<1)||(cols<1)){
	return;
     }

//      clock_t start = clock();
     std::time_t start, end;
     std::time(&start);	

     MultiArray<2, float> features(Shape(rows,cols));
     MultiArray<2, int> labels(Shape(rows,1));

     int numzeros=0; 	
     for(int i=0; i < rows; i++){
	 labels(i,0) = plabels[i];
	 numzeros += (labels(i,0)==-1?1:0);
	 for(int j=0; j < cols ; j++){
	     features(i,j) = (float)pfeatures[i][j];	
	 }
     }
     printf("Number of merge: %d\n",numzeros);



     printf("Number of trees:  %d\n",_tree_count);
     RandomForestOptions rfoptions = RandomForestOptions().tree_count(_tree_count).use_stratification(RF_EQUAL);	//RF_EQUAL, RF_PROPORTIONAL
//      RandomForestOptions rfoptions = RandomForestOptions().tree_count(_tree_count).use_stratification(RF_EQUAL).min_split_node_size(10);	//RF_EQUAL, RF_PROPORTIONAL
     _rf= new RandomForest<>(rfoptions);


     // construct visitor to calculate out-of-bag error
     visitors::OOB_Error oob_v;
     visitors::VariableImportanceVisitor varimp_v;

     _rf->learn(features, labels);
     _nfeatures = _rf->column_count();
     _nclass = _rf->class_count();
     int rntrees= _rf->tree_count();

     std::time(&end);
     printf("Time required to learn RF: %.2f sec with %d trees for %d classes\n", (difftime(end,start))*1.0, rntrees, _nclass);
     float ncorr = 0;
     for(size_t i=0; i<pfeatures.size();i++){
	  float pp = predict(pfeatures[i]);
	  ncorr += ((pp<0.5 && plabels[i]==1)?1:0);
	  ncorr += ((pp>=0.5 && plabels[i]==-1)?1:0);
     }
       
     printf("with oob :%f\n", ncorr/pfeatures.size());
}

void VigraRFclassifier::save_classifier(const char* rf_filename){
     HDF5File rf_file(rf_filename,HDF5File::New); 	
     rf_export_HDF5(*_rf, rf_file,"rf");
}

