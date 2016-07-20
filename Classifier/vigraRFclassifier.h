
#ifndef _vigra_rf_classifier
#define _vigra_rf_classifier

#include <vigra/multi_array.hxx>
#include <vigra/random_forest.hxx>
#include <vigra/random_forest/rf_common.hxx>
#include <vigra/hdf5impex.hxx>
#include <vigra/random_forest_hdf5_impex.hxx>

#include "edgeclassifier.h"

using namespace std;
using namespace vigra;
using namespace rf;

typedef MultiArray<2, float>::difference_type Shape;

class VigraRFclassifier: public EdgeClassifier{


     RandomForest<>* _rf;
     int _nfeatures;
     int _nclass;	
	
     int _tree_count; 	

public:
     VigraRFclassifier():_rf(NULL), _tree_count(255) {};	
     VigraRFclassifier(int ptree_count):_rf(NULL), _tree_count(ptree_count) {};	
     VigraRFclassifier(const char* rf_filename);
     ~VigraRFclassifier(){
	 if (_rf) delete _rf;
     }	
     void  load_classifier(const char* rf_filename);
     float predict(std::vector<float>& features);
     void predict_m(std::vector<float>& features, std::vector<float>& class_prob);
     void learn(std::vector< std::vector<float> >& pfeatures, std::vector<int>& plabels);
     void save_classifier(const char* rf_filename);
     bool is_trained(){
	if (_rf && _rf->tree_count()>0)
	   return true;
	else 
	   return false;
		
     };

};

#endif
