
#ifndef _opencv_rf_classifier
#define _opencv_rf_classifier

#include "opencv/ml.h"

#include "edgeclassifier.h"

using namespace std;


class OpencvRFclassifier: public EdgeClassifier{

    CvRTrees* _rf;	
    int _nfeatures;
    int _nclass;	

    vector<CvForestTree*> _trees;

    vector<float> _tree_weights;	

    bool _use_tree_weights; 

    int _tree_count;
    int _max_depth;	
		
    	

public:
     OpencvRFclassifier():_rf(NULL), _tree_count(255), _max_depth(20) {};	
     OpencvRFclassifier(int ptree_count, int pmax_depth):_rf(NULL), _tree_count(ptree_count), _max_depth(pmax_depth) {};	
     OpencvRFclassifier(const char* rf_filename);
     ~OpencvRFclassifier(){
	 if (_rf) delete _rf;
     }	
     void  load_classifier(const char* rf_filename);
     float predict(std::vector<float>& features);
     void learn(std::vector< std::vector<float> >& pfeatures, std::vector<int>& plabels);
     void save_classifier(const char* rf_filename);

     void set_tree_weights(vector<float>& pwts);	
     void get_tree_responses(vector<float>& pfeatures,vector<float>& responses);	
     void reduce_trees();	

     bool is_trained(){
	if (_rf && _rf->get_tree_count()>0)
	   return true;
	else 
	   return false;
		
     };

};

#endif
