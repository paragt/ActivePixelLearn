
#ifndef _edge_classifier
#define _edge_classifier

class EdgeClassifier{


public: 
	virtual void load_classifier(const char*)=0;
	virtual float predict(std::vector<float>&){

  	    //srand ( time(NULL) );
            //float val= rand()*(1.0/ RAND_MAX);
            float val= 0.5;
	    return val;	
	    
	}
	virtual void predict_m(std::vector<float>& features, std::vector<float>& class_prob)=0;
	virtual void learn(std::vector< std::vector<float> >& pfeatures, std::vector<int>& plabels)=0;
	virtual void save_classifier(const char* rf_filename)=0;
	virtual bool is_trained()=0;

        virtual ~EdgeClassifier() {}


     	virtual void set_tree_weights(std::vector<float>& pwts){};	
	virtual void get_tree_responses(std::vector<float>& pfeatures,std::vector<float>& responses){};
	virtual void reduce_trees(){};
};


#endif
