
#ifndef _PIXEL_DETECTOR
#define _PIXEL_DETECTOR


#include <vector>
#include <set>

#include "Utilities/h5read.h"
#include "Utilities/h5write.h"
#include "Utilities/Dataset.h"
#include "SemiSupervised/weightmatrix_iter.h"
// #include "SemiSupervised/weightmatrix1.h"
#include "Classifier/opencvRFclassifier.h"
#include "Classifier/vigraRFclassifier.h"
#include "SemiSupervised/kmeans.h"


#include <iostream>
#include <memory>

#include <ctime>
#include <cmath>
#include <cstring>

using namespace std;

typedef std::map<unsigned int, float> LabelList;
typedef std::map<unsigned int, vector<float> > ClassProb;
typedef std::map<unsigned int, vector<float> >::iterator ClassProbIterator;


template<typename T>
T vec_dist( vector<T>& vec1, vector<T>& vec2) {
    T ret=0;
    
    for(size_t i=0; i< vec1.size(); i++)
	ret += (vec1[i]-vec2[i])*(vec1[i]-vec2[i]);
    
    return ret;
}


class PixelDetector{
  
    int m_depth, m_width, m_height, m_nfeat;
    
    Dataset m_dtst_all;
    Dataset m_dtst_ds;
    
    vector< WeightMatrix_iter* > m_wtmat;
//     vector< WeightMatrix1* > m_wtmat;


    
    vector<EdgeClassifier*> m_pclfrs;
    EdgeClassifier* m_clfr_mult;
    
    size_t m_nclass;
    size_t m_ntrees;
    
    float* m_feature_data;
    int* m_groundtruth_data;
    int* m_mask_data;

    vector< vector<unsigned int> > m_gen_idx;

    ClassProb m_class_prob_gen;
    ClassProb m_class_prob_dis;
    
    vector <float> m_class_bias;
    vector <float> m_prior_belief;
    
    size_t m_init_sz;
    
    std::map<unsigned int, int> m_all_unique_idx;
    vector< vector<unsigned int> > m_dis_idx;
    vector< vector<unsigned int> > m_mostrecent_idx;

    std::vector<unsigned int> m_all_cidx;
    
    unsigned int m_count_low_disagree;

public:
  
    PixelDetector(unsigned int pdepth, unsigned int pheight, unsigned int pwidth,
		  unsigned int pnfeat, float* ppred, int* pgt, unsigned int pnclass, int pntrees){
	
	m_depth = pdepth;
	m_height = pheight;
	m_width = pwidth;
	m_nfeat = pnfeat;
	
	m_feature_data = ppred;
	m_groundtruth_data = pgt;
      
	m_nclass = pnclass;
	m_ntrees=pntrees;
	unsigned int npairs = m_nclass*(m_nclass-1)/2;
	m_pclfrs.resize(npairs);
	for (size_t i=0; i < npairs; i++)
	    m_pclfrs[i] = new VigraRFclassifier(m_ntrees);
	
	m_clfr_mult = new VigraRFclassifier(m_ntrees);
	
	m_class_bias.resize(m_nclass);
	m_prior_belief.resize(m_nclass);
	for(size_t i=0; i<m_nclass; i++){
	    m_class_bias[i] = 1.0;
	    m_prior_belief[i] = 1.0;
	}
	m_prior_belief[1]= 1.25;

	m_wtmat.clear();
	m_wtmat.resize(npairs);	
	for (size_t i=0; i < npairs; i++)
	    m_wtmat[i] = NULL;
	
	m_count_low_disagree = 0;
	
	m_mask_data = NULL;
    }
    ~PixelDetector(){
	int rmp=1;
	unsigned int npairs = m_nclass*(m_nclass-1)/2;
	for (size_t i=0; i < npairs; i++)
	    if(m_wtmat[i] != NULL)
		delete m_wtmat[i];
      
	for (size_t i=0; i < npairs; i++)
	    delete m_pclfrs[i];
	
	delete m_clfr_mult;  
    };
//     void read_data(unsigned int*, unsigned int*);
    void read_data();
    void build_wtmat(bool read_off, string feature_filename, double wt_thd);
    void set_code_idx();
    void select_initial_points();
    void select_initial_points2();
    void select_initial_points_biased();
    void select_initial_points_fixed();
    void select_initial_points_kmeans();
    void add_extra_samples(size_t );
    
    void propagate_labels();
    void propagate_labels_multiclass();
    void propagate_labels_classifier();
    void get_random_stpoints2(int label, unsigned int len, vector<unsigned int>& idx);
    void get_random_stpoints(int label, unsigned int len, vector<unsigned int>& idx);
    void compute_confusion(ClassProb& final_prob);
    void convert2multiclass_prob(std::vector< std::vector <float> >& preds, 
				 std::vector< std::vector <int> >& pair_ids, ClassProb& final_prob);
    void learn_classifier_pairwise();
    void learn_classifier_multiclass();
    void predict_pairwise();
    void predict_multiclass();
    
    void multiclass_probability(int k, float **r, float *p);
    void save_classifier(string clfr_name);
    void save_classifier_inter(string clfr_name);
    void save_classifier_m(string clfr_name);
    void save_classifier_inter_m(string clfr_name);
  
    float measure_info(std::vector<float>& gprob, std::vector<float>& dprob);
    float measure_info_old(std::vector<float>& gprob, std::vector<float>& dprob);
    float measure_info_boundary(std::vector<float>& gprob, std::vector<float>& dprob);
    float measure_info_margin(std::vector<float>& gprob, std::vector<float>& dprob);
    void find_most_informative(std::multimap<float, unsigned int>&  info_list);
    bool add_new_points(unsigned int eta);
    void predict_pairwise_all(float *predictions, unsigned int pred_stidx, unsigned int pred_enidx=0);
    void predict_multiclass_all(float *predictions, unsigned int pred_stidx, unsigned int pred_enidx=0);
    void load_classifier(string clfr_name);
    void load_classifier_m(string clfr_name);
    
    size_t trnset_size();
    void trnset_size_class();
    
    void set_initsz_class(size_t psz){m_init_sz = psz;};
//     void solve_linear(vector<unsigned int>& new_idx, vector<int>& new_lbl, WeightMatrix1*, LabelList& prop_lbl);
    int label_from_prob(vector<float>& prob);
    
    void recompute_class_bias();
    void recompute_class_bias2();
    void recompute_class_bias3();
    void bootstrap_samples(vector< vector<unsigned int> >& input_idx,
				      vector< vector<unsigned int> >& bootstrap_idx);

    void set_classifier(PixelDetector* pd){
	this->m_clfr_mult = pd->m_clfr_mult;
    }
    void downsample(string& ctridx_filename, size_t ndata);
    void downsample(size_t ndata, std::vector<unsigned int> offset);
    
    void set_mask(int* pmask){ m_mask_data = pmask;}
}; 


#endif
