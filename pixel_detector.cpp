#include "pixel_detector.h"


#define EPS 0.0001

void PixelDetector::read_data(){


    // 	float* feature_single_ch = new float[m_depth*m_height*m_width];
    // 	float* feature_ch0 = new float[m_depth*m_height*m_width];


    size_t cube_size, plane_size;
    size_t position, position_l, count, i, j, k;		    	
    vector<float> feat0(m_nfeat, 0.0);
    count = 0;
    
    vector< vector<float> >& all_features = m_dtst_all.get_features();
    vector<int>& all_labels =  m_dtst_all.get_labels();
    float dist_thd=0.5;
    
    all_features.clear();
    all_labels.clear();
    vector<unsigned int> count_lbl_all(m_nclass,0);
//     all_labels.resize(all_features.size());
    for(i=0; i<m_depth; i++){
	cube_size = m_height*m_width*m_nfeat;	
	for(j=0; j<m_height; j++){
	    plane_size = m_width*m_nfeat;
	    for(k=0; k<m_width; k++){
	        
		position = i*cube_size + j*plane_size + k*m_nfeat ;
		memcpy((void*)(feat0.data()), (void*) (m_feature_data+position), m_nfeat*sizeof(float));
		all_features.push_back(feat0);
		
		
		if (m_groundtruth_data){
		    position_l = i*(cube_size/m_nfeat) + j*(plane_size/m_nfeat) + k ;
		    all_labels.push_back(m_groundtruth_data[position_l]);
		    (count_lbl_all[m_groundtruth_data[position_l]])++;
		}
		
// 		feature0[position_l] = m_feature_data[position_l];
// 		feature1[position_l] = m_feature_data[position_l+1];
		
	    }		
	}	
    }
  
    m_dtst_all.initialize();
   
    /*C* debug
    FILE* feat_fp = fopen("test_feat.txt","wt");
    FILE* gt_fp = fopen("test_gt.txt","wt");
    for(i=10; i<11; i++){
	cube_size = m_height*m_width*m_nfeat;	
	for(j=0; j<m_height; j++){
	    plane_size = m_width*m_nfeat;
	    for(k=0; k<m_width; k++){
	        
		position = i*cube_size + j*plane_size + k*m_nfeat ;
		fprintf(feat_fp, "%f\n", m_feature_data[position+1]); //1st feature
		
		if (m_groundtruth_data){
		    position_l = i*(cube_size/m_nfeat) + j*(plane_size/m_nfeat) + k ;
		    fprintf(gt_fp,"%d\n",m_groundtruth_data[position_l]);
		}
		
		
	    }		
	}	
    }
    fclose(feat_fp);
    fclose(gt_fp);
    /**/
    
//     for(size_t tmpi=0; tmpi<all_features.size(); tmpi++){
// 	feat_ret[tmpi] = (unsigned int ) (all_features[tmpi][1]);
// 	gt_ret[tmpi] = (unsigned int ) (all_labels[tmpi]);
//     }
    
    
}
void PixelDetector::downsample( size_t ndata, std::vector<unsigned int> offset){
  
    for(size_t i= offset[0]; i < (m_depth-offset[0]); i++){
	size_t plane_size = m_height*m_width;	
	for(size_t j = offset[1]; j < (m_height-offset[1]); j+=2){
	    for(size_t k = offset[2]; k < (m_width-offset[2]) ; k+=2){
		size_t position_l = i*plane_size + j*m_width + k ;
		if(m_mask_data){ 
		  if(m_mask_data[position_l])
		    m_all_cidx.push_back(position_l);
		}
		else
		    m_all_cidx.push_back(position_l);
		  
// 		all_labels.push_back(m_groundtruth_data[position_l]);
	    }
	}
    }
    printf("Data size is %u from a (%u, %u, %u) volume\n", m_all_cidx.size(), m_depth, m_height, m_width);
    
    random_shuffle(m_all_cidx.begin(), m_all_cidx.end());
  
    m_all_cidx.erase(m_all_cidx.begin()+ndata, m_all_cidx.end());    
    
    set_code_idx();
  
}

void PixelDetector::downsample(string& ctridx_filename, size_t ndata)
{
    m_all_cidx.clear();
    FILE* fp = fopen(ctridx_filename.c_str(),"rt");
    unsigned int aa;
    while(!feof(fp)){
	fscanf(fp,"%u\n",&aa);
	m_all_cidx.push_back(aa);
    }
    fclose(fp);
    
    m_all_cidx.erase(m_all_cidx.begin()+ndata, m_all_cidx.end());    
    
    set_code_idx();
    
}

void PixelDetector::set_code_idx()
{
    
    vector< vector<float> >& all_features = m_dtst_all.get_features();
    vector<int>& all_labels =  m_dtst_all.get_labels();
    
//     vector< vector<float> > scaled_features;
//     wtmat.scale_features(all_features, scaled_features); ;
    
    vector< vector<float> >& ds_features = m_dtst_ds.get_features();
    vector<int>& ds_labels =  m_dtst_ds.get_labels();
//     vector< vector<float> > scaled_features_ds;
    
    vector<unsigned int> count_lbl(m_nclass,0);
    for(size_t i=0; i< m_all_cidx.size(); i++){
	unsigned int idx = m_all_cidx[i];
	ds_features.push_back(all_features[idx]);
	ds_labels.push_back(all_labels[idx]);
	(count_lbl[all_labels[idx]])++;
// 	scaled_features_ds.push_back(scaled_features[idx]);
    }
    m_dtst_ds.initialize();
    
    for(int i=0; i<m_nclass; i++)
      printf("class %d = %u, ",i, count_lbl[i]); 
    printf("\n");
    
}
size_t PixelDetector::trnset_size(){
    return m_all_unique_idx.size();
//     size_t total=0;
//     
//     for(int i=0; i<m_gen_idx.size(); i++)
// 	total += (m_gen_idx[i].size());
//     
//     return total;
}

void PixelDetector::trnset_size_class()
{
    vector<size_t> class_total(m_nclass,0);
    for(int i=0; i<m_nclass; i++){
	class_total[i] = m_dis_idx[i].size();
	printf("class %d = %u, ",i, class_total[i]); 
    }
    printf("\n");
}


// void PixelDetector::solve_linear(vector<unsigned int>& new_idx, vector<int>& new_lbl, WeightMatrix1* wtmat, LabelList& prop_lbl){
//     std::time_t start, end;
//     std::time(&start);	
//     wtmat->add2trnset(new_idx, new_lbl);
// //     wtmat->solve(prop_lbl);
//     wtmat->AMGsolve(prop_lbl);
//     std::time(&end);	
//     printf("C: Time to solve linear equations: %.2f sec\n", (difftime(end,start))*1.0);
// }

void PixelDetector::propagate_labels()
{
    vector<int>& ds_labels =  m_dtst_ds.get_labels();  
    unsigned int npairs = m_nclass*(m_nclass-1)/2;
    
    std::vector< std::vector <float> > preds(npairs);
    std::vector< std::vector <int> > pair_ids(npairs);
    for(size_t i=0; i< npairs; i++){
	preds[i].resize(ds_labels.size(),-1);
	pair_ids[i].resize(2);
    }
	
    
    size_t ipair=0;
    size_t maxsz = 0, maxid;
    std::map<unsigned int, float>::iterator pit;
    
    vector<LabelList> prop_lbl(npairs);
    std::vector< boost::thread* > threads;
    vector< vector<unsigned int> > new_idx(npairs);
    vector<vector<int> > new_lbl(npairs);
    for(size_t i=0; i < m_nclass-1; i++){
	for(size_t j=i+1; j< m_nclass ; j++, ipair++ ){
	    
	    new_idx[ipair] = m_gen_idx[i];
	    new_lbl[ipair].clear();
	    new_lbl[ipair].resize(new_idx[ipair].size(), 1);
	    
	    pair_ids[ipair][0] = i; pair_ids[ipair][1] = j;
	    
	    new_idx[ipair].insert(new_idx[ipair].end(), m_gen_idx[j].begin(), m_gen_idx[j].end());
	    new_lbl[ipair].insert(new_lbl[ipair].end(), m_gen_idx[j].size(), -1);
	    
// // // 	    threads.push_back(new boost::thread(&PixelDetector::solve_linear, this, 
// // // 					 new_idx[ipair],new_lbl[ipair], m_wtmat[ipair],
// // // 						boost::ref(prop_lbl[ipair])));
// 	    new_idx[ipair] = m_mostrecent_idx[i];
// 	    new_lbl[ipair].clear();
// 	    new_lbl[ipair].resize(new_idx[ipair].size(), 1);
// 	    
// 	    pair_ids[ipair][0] = i; pair_ids[ipair][1] = j;
// 	    
// 	    new_idx[ipair].insert(new_idx[ipair].end(), m_mostrecent_idx[j].begin(), m_mostrecent_idx[j].end());
// 	    new_lbl[ipair].insert(new_lbl[ipair].end(), m_mostrecent_idx[j].size(), -1);
// 	    
// 	    threads.push_back(new boost::thread(&PixelDetector::solve_linear, this, 
// 					 new_idx[ipair],new_lbl[ipair], m_wtmat[ipair],
// 						boost::ref(prop_lbl[ipair])));
	}
	    
    }
    printf("Sync all threads \n");
    for (size_t ti=0; ti<threads.size(); ti++) 
      (threads[ti])->join();
    printf("all threads done\n");
    
    for(ipair=0; ipair < npairs; ipair++){
	for(pit = prop_lbl[ipair].begin(); pit != prop_lbl[ipair].end(); pit++){
	    unsigned int idx = pit->first;
	    float tmp_pred = pit->second;
	    float deg = m_wtmat[0]->get_degree(idx);
// 	    float sqrtdeg = float(sqrt(deg));
// 	    float val_1 = tmp_pred*sqrtdeg;
	    
	    float val_1 = tmp_pred;

	    float val = 0.5*(val_1+1);
	    
	    val = (val >= 1.0)? 1.0  : val;
	    val = (val <= 0.0)? 0.0 : val;
	    
	    preds[ipair][pit->first] = val;
	}
    }

    m_class_prob_gen.clear();
    convert2multiclass_prob(preds, pair_ids, m_class_prob_gen);
    
    /*C* debug
    float eps1=1e-9;
    unsigned int nconn=0, nfound=0;;
    ClassProbIterator cit = m_class_prob_gen.begin();
    for(; cit!= m_class_prob_gen.end(); cit++){
	vector<float>& tmpvec = cit->second;
	if ( (fabs(tmpvec[0] - tmpvec[1]) < eps1) && (fabs(tmpvec[1] - tmpvec[2]) < eps1)
	  && (fabs(tmpvec[2] - tmpvec[0]) < eps1) ){
	    nconn += m_wtmat[0]->num_connections(cit->first) ; 
	    nfound++;
	}
    }
    printf("found %u times, #connections: %u\n", nfound, nconn);
    /**/
    printf("LabelProp: total predicted: %u\n",m_class_prob_gen.size());
    compute_confusion(m_class_prob_gen);  
    
    
    
//     std::map<unsigned int, float>::iterator pit;
//     unsigned int count=0;
//     for(pit = prop_lbl.begin(); pit!=prop_lbl.end(); pit++)
// 	if (pit->second>0)
// 	    count++;
// 	
//     printf("total positive labels: %u\n",count);
    
    
}

void PixelDetector::propagate_labels_classifier()
{
    
//     vector<unsigned int> new_idx_multiclass;
//     vector<int> new_lbl_multiclass;
//     
//     for(size_t i=0; i < m_nclass; i++){
// 	new_idx_multiclass.insert(new_idx_multiclass.end(), m_mostrecent_idx[i].begin(), m_mostrecent_idx[i].end());
// 	new_lbl_multiclass.insert(new_lbl_multiclass.end(), m_mostrecent_idx[i].size(), i);
//     }    
    printf("Generative: ");
    for(int i=0; i<m_nclass; i++)
      printf("class %d = %u, ",i, m_gen_idx[i].size()); 
    printf("\n");

    vector<int>& ds_labels =  m_dtst_ds.get_labels();  
    vector<unsigned int> trn_idx;
    
	
    vector< vector<float> > cum_train_features;
    vector<int> cum_train_labels ;
    
    size_t ipair=0;
    size_t maxsz = 0, maxid;
    
    vector<unsigned int> new_idx;
    vector<int> new_lbl;
    
    for(size_t i=0; i < m_nclass; i++){
	new_idx.insert(new_idx.end(), m_gen_idx[i].begin(), m_gen_idx[i].end());
	vector<unsigned int>& tmp_idx = m_gen_idx[i];
	for(size_t j=0; j< tmp_idx.size(); j++){
	    int lbl = ds_labels[tmp_idx[j]];
	    new_lbl.push_back(lbl);
	}
    }
    
	    
    m_dtst_ds.clear_trn_labels();
    trn_idx.clear();
    
    trn_idx.insert(trn_idx.end(), new_idx.begin(), new_idx.end());
    m_dtst_ds.append_trn_labels(new_lbl);
    m_dtst_ds.get_train_data(trn_idx, cum_train_features, cum_train_labels);
    m_pclfrs[0]->learn(cum_train_features, cum_train_labels);
    
    
// //     std::time_t start, end;
// //     std::time(&start);	
// //     m_wtmat[0]->add2trnset(new_idx_multiclass, new_lbl_multiclass);
// // //     wtmat->solve(prop_lbl);
// //     m_wtmat[0]->solve(m_class_prob_gen);
// //     std::time(&end);	
// //     printf("C: Time to propagate: %.2f sec\n", (difftime(end,start))*1.0);
// //     
// //     std::vector<float> prior_prob(m_nclass,0.0);
// //     std::vector<float> class_mass(m_nclass,0.0);
// //     
// //     float total_gen=0;

//     for(int cc=0; cc<m_nclass; cc++){
// 	(prior_prob[cc]) += m_gen_idx[cc].size();
// 	total_gen += m_gen_idx[cc].size();
//     }
    
    /*C* debug
    float eps1=1e-9;
    unsigned int nconn=0, nfound=0;;
    ClassProbIterator cit = m_class_prob_gen.begin();
    for(; cit!= m_class_prob_gen.end(); cit++){
	vector<float>& tmpvec = cit->second;
	if ( (fabs(tmpvec[0] - tmpvec[1]) < eps1) && (fabs(tmpvec[1] - tmpvec[2]) < eps1)
	  && (fabs(tmpvec[2] - tmpvec[0]) < eps1) ){
	    nconn += m_wtmat[0]->num_connections(cit->first) ; 
	    nfound++;
	}
	
// 	for(int cc=0; cc<m_nclass; cc++){
// 	    (class_mass[cc]) += tmpvec[cc];
// 	}
    }
    printf("found %u times, #connections: %u\n", nfound, nconn);
    /**/
//     for(cit = m_class_prob_gen.begin(); cit!= m_class_prob_gen.end(); cit++){
// 	vector<float>& tmpvec = cit->second;
// 	for(int cc=0; cc<m_nclass; cc++){
// 	    tmpvec[cc] *= ((prior_prob[cc]*m_class_prob_gen.size())*1.0/(class_mass[cc]*total_gen));
// 	}
//     }
    
    vector< vector<float> >& ds_features = m_dtst_ds.get_features();
//     vector<int>& ds_labels =  m_dtst_ds.get_labels();  
    
    m_class_prob_gen.clear();
    vector<float> predp_prob(m_nclass);
    for(unsigned int dd=0; dd< ds_features.size(); dd++){
      
	if (m_all_unique_idx.find(dd)!=m_all_unique_idx.end()) // already in trn set
	    continue;

	m_pclfrs[0]->predict_m(ds_features[dd], predp_prob);
	m_class_prob_gen.insert(std::make_pair(dd,predp_prob));
    }
    
    printf("LabelProp: total predicted: %u\n",m_class_prob_gen.size());
    compute_confusion(m_class_prob_gen);  
    
    
    
//     std::map<unsigned int, float>::iterator pit;
//     unsigned int count=0;
//     for(pit = prop_lbl.begin(); pit!=prop_lbl.end(); pit++)
// 	if (pit->second>0)
// 	    count++;
// 	
//     printf("total positive labels: %u\n",count);
    
    
}


void PixelDetector::propagate_labels_multiclass()
{
    
    vector<unsigned int> new_idx_multiclass;
    vector<int> new_lbl_multiclass;
    
    for(size_t i=0; i < m_nclass; i++){
	new_idx_multiclass.insert(new_idx_multiclass.end(), m_mostrecent_idx[i].begin(), m_mostrecent_idx[i].end());
	new_lbl_multiclass.insert(new_lbl_multiclass.end(), m_mostrecent_idx[i].size(), i);
    }    
    printf("Generative: ");
    for(int i=0; i<m_nclass; i++)
      printf("class %d = %u, ",i, m_gen_idx[i].size()); 
    printf("\n");
    
    std::time_t start, end;
    std::time(&start);	
    m_wtmat[0]->add2trnset(new_idx_multiclass, new_lbl_multiclass);
//     wtmat->solve(prop_lbl);
    m_wtmat[0]->solve(m_class_prob_gen);
    std::time(&end);	
    printf("C: Time to propagate: %.2f sec\n", (difftime(end,start))*1.0);
    
    std::vector<float> prior_prob(m_nclass,0.0);
    std::vector<float> class_mass(m_nclass,0.0);
    
    float total_gen=0;
//     for(int cc=0; cc<m_nclass; cc++){
// 	(prior_prob[cc]) += m_gen_idx[cc].size();
// 	total_gen += m_gen_idx[cc].size();
//     }
    
    /*C* debug
    float eps1=1e-9;
    unsigned int nconn=0, nfound=0;;
    ClassProbIterator cit = m_class_prob_gen.begin();
    for(; cit!= m_class_prob_gen.end(); cit++){
	vector<float>& tmpvec = cit->second;
	if ( (fabs(tmpvec[0] - tmpvec[1]) < eps1) && (fabs(tmpvec[1] - tmpvec[2]) < eps1)
	  && (fabs(tmpvec[2] - tmpvec[0]) < eps1) ){
	    nconn += m_wtmat[0]->num_connections(cit->first) ; 
	    nfound++;
	}
	
// 	for(int cc=0; cc<m_nclass; cc++){
// 	    (class_mass[cc]) += tmpvec[cc];
// 	}
    }
    printf("found %u times, #connections: %u\n", nfound, nconn);
    /**/
//     for(cit = m_class_prob_gen.begin(); cit!= m_class_prob_gen.end(); cit++){
// 	vector<float>& tmpvec = cit->second;
// 	for(int cc=0; cc<m_nclass; cc++){
// 	    tmpvec[cc] *= ((prior_prob[cc]*m_class_prob_gen.size())*1.0/(class_mass[cc]*total_gen));
// 	}
//     }
    
    
    printf("LabelProp: total predicted: %u\n",m_class_prob_gen.size());
    compute_confusion(m_class_prob_gen);  
    
    
    
//     std::map<unsigned int, float>::iterator pit;
//     unsigned int count=0;
//     for(pit = prop_lbl.begin(); pit!=prop_lbl.end(); pit++)
// 	if (pit->second>0)
// 	    count++;
// 	
//     printf("total positive labels: %u\n",count);
    
    
}

void PixelDetector::predict_pairwise_all(float *predictions, 
					unsigned int pred_stidx, unsigned int pred_enidx)
{
    vector< vector<float> >& ds_features = m_dtst_all.get_features();
    unsigned int npairs = m_nclass*(m_nclass-1)/2;
    
    std::vector< std::vector <float> > preds(npairs);
    std::vector< std::vector <int> > pair_ids(npairs);
    for(size_t i=0; i< npairs; i++){
	preds[i].resize(ds_features.size(),-1);
	pair_ids[i].resize(2);
    }
	
    
    size_t ipair;
    size_t maxsz = 0, maxid;
    std::map<unsigned int, float>::iterator pit;
    
    if (pred_enidx==0){
	pred_stidx = 0;
	pred_enidx = ds_features.size();
    }
    
    for(size_t dd= pred_stidx; dd< pred_enidx; dd++){
	ipair=0;
	for(size_t i=0; i < m_nclass-1; i++){
	    for(size_t j=i+1; j<m_nclass; j++, ipair++ ){
		
		
		pair_ids[ipair][0] = i; pair_ids[ipair][1] = j;
		
		
		float pred1 = m_pclfrs[ipair]->predict(ds_features[dd]); // number of trees
		pred1 = (pred1==1)? pred1-EPS : pred1;
		pred1 = (pred1==0)? EPS : pred1;
		
		preds[ipair][dd] = pred1;
		  
	    }
	}
    }
    
    m_class_prob_dis.clear();
    convert2multiclass_prob(preds, pair_ids, m_class_prob_dis);
    ClassProbIterator cit = m_class_prob_dis.begin();
    size_t pos=0;
    for(; cit!=m_class_prob_dis.end(); cit++){
	std::vector<float>& prob1 = cit->second;
	for(size_t cc=0; cc<prob1.size(); cc++)	
	    predictions[pos++] = prob1[cc];
    }

    
}

void PixelDetector::predict_multiclass_all(float *predictions, 
					   unsigned int pred_stidx, unsigned int pred_enidx)
{
    vector< vector<float> >& ds_features = m_dtst_all.get_features();
    
    vector<float> predp_prob(m_nclass);
    size_t pos=0;
    
    if (pred_enidx==0){
	pred_stidx = 0;
	pred_enidx = ds_features.size();
    }
      
    for(unsigned int dd=pred_stidx; dd< pred_enidx; dd++){

	m_clfr_mult->predict_m(ds_features[dd], predp_prob);
	for(size_t cc=0; cc< predp_prob.size(); cc++)	
	    predictions[pos++] =  predp_prob[cc];
    }

}


void PixelDetector::predict_pairwise()
{
    vector< vector<float> >& ds_features = m_dtst_ds.get_features();
    vector<int>& ds_labels =  m_dtst_ds.get_labels();  
    unsigned int npairs = m_nclass*(m_nclass-1)/2;
    
    std::vector< std::vector <float> > preds(npairs);
    std::vector< std::vector <int> > pair_ids(npairs);
    for(size_t i=0; i< npairs; i++){
	preds[i].resize(ds_labels.size(),-1);
	pair_ids[i].resize(2);
    }
	
    
    size_t ipair;
    size_t maxsz = 0, maxid;
    std::map<unsigned int, float>::iterator pit;
    for(size_t dd=0; dd< ds_features.size(); dd++){
      
	if (m_all_unique_idx.find(dd)!=m_all_unique_idx.end()) // already in trn set
	    continue;
	  
	ipair=0;
	for(size_t i=0; i < m_nclass-1; i++){
	    for(size_t j=i+1; j<m_nclass; j++, ipair++ ){
		
		
		pair_ids[ipair][0] = i; pair_ids[ipair][1] = j;
		
		
		float pred1 = m_pclfrs[ipair]->predict(ds_features[dd]); // number of trees
// 		pred1 = (pred1==1)? pred1-0.001:pred1;
// 		pred1 = (pred1==0)? 0.001:pred1;
		
		preds[ipair][dd] = pred1;
		  
	    }
	}
    }
    
    m_class_prob_dis.clear();
    convert2multiclass_prob(preds, pair_ids, m_class_prob_dis);

    compute_confusion(m_class_prob_dis);  
    
}
void PixelDetector::predict_multiclass()
{
    vector< vector<float> >& ds_features = m_dtst_ds.get_features();
    vector<int>& ds_labels =  m_dtst_ds.get_labels();  
    
    size_t ipair;
    size_t maxsz = 0, maxid;
    m_class_prob_dis.clear();
    vector<float> predp_prob(m_nclass);
    for(unsigned int dd=0; dd< ds_features.size(); dd++){
      
	if (m_all_unique_idx.find(dd)!=m_all_unique_idx.end()) // already in trn set
	    continue;

	m_clfr_mult->predict_m(ds_features[dd], predp_prob);
	m_class_prob_dis.insert(std::make_pair(dd,predp_prob));
    }

    
    /*C* debug
    ClassProbIterator cit = m_class_prob_dis.begin();
    for(; cit!= m_class_prob_dis.end(); cit++){
	vector<float>& tmpvec = cit->second;
	if (tmpvec[0]==1 && tmpvec[1]==0 && tmpvec[2]==0){
	    int pp=1;
	}
    }
    /**/
    compute_confusion(m_class_prob_dis);  
    
}

void PixelDetector::learn_classifier_pairwise()
{
    vector<int>& ds_labels =  m_dtst_ds.get_labels();  
    unsigned int npairs = m_nclass*(m_nclass-1)/2;
     vector<unsigned int> trn_idx;
   
	
    vector< vector<float> > cum_train_features;
    vector<int> cum_train_labels ;
    
    size_t ipair=0;
    size_t maxsz = 0, maxid;
    std::map<unsigned int, float>::iterator pit;
    for(size_t i=0; i < m_nclass-1; i++){
	for(size_t j=i+1; j<m_nclass; j++, ipair++ ){
	    
	    m_dtst_ds.clear_trn_labels();
	    trn_idx.clear();
	    
	    vector<unsigned int> new_idx = m_dis_idx[i];
	    vector<int> new_lbl(new_idx.size(), 1);
	    
	    
	    new_idx.insert(new_idx.end(), m_dis_idx[j].begin(), m_dis_idx[j].end());
	    new_lbl.insert(new_lbl.end(), m_dis_idx[j].size(), -1);

	    
	    trn_idx.insert(trn_idx.end(), new_idx.begin(), new_idx.end());
	    m_dtst_ds.append_trn_labels(new_lbl);
	    m_dtst_ds.get_train_data(trn_idx, cum_train_features, cum_train_labels);
	    
	    m_pclfrs[ipair]->learn(cum_train_features, cum_train_labels); // number of trees
	      
	}
    }
    
    
}

void PixelDetector::bootstrap_samples(vector< vector<unsigned int> >& input_idx,
				      vector< vector<unsigned int> >& bootstrap_idx)

{
    bootstrap_idx.clear(); bootstrap_idx.resize(m_nclass);
    for(size_t i=0; i<m_nclass ; i++){
	vector<unsigned int> index_c = input_idx[i];
	unsigned int idx_sz = index_c.size();
	unsigned int bootstrap_sz = (unsigned int)(idx_sz*m_class_bias[i]);
	
	bootstrap_idx[i].clear();
	
	if (bootstrap_sz > idx_sz){
	    set<unsigned int> bsamples_unique;
	    for(size_t j=0; j< idx_sz ;j++){
		unsigned int idx1 = index_c[j]; 
		bootstrap_idx[i].push_back(idx1);
		bsamples_unique.insert(idx1);
	    }
	    vector<unsigned int> extra_samples;
	    get_random_stpoints2(i, (bootstrap_sz - idx_sz)*10 , extra_samples); 
// 	    bootstrap_idx[i].insert(bootstrap_idx[i].end(), extra_samples.begin(), extra_samples.end());
	    size_t ll=0;
	    while (bootstrap_idx[i].size()<bootstrap_sz){
		if (ll==extra_samples.size())
		    break;
		if(bsamples_unique.find(extra_samples[ll]) == bsamples_unique.end()){
		    bootstrap_idx[i].push_back(extra_samples[ll]);
		    bsamples_unique.insert(extra_samples[ll]);
		}
		ll++;
	    }
	}
	else{
	    for(size_t j=0; j< bootstrap_sz ;j++){
		unsigned int idx1 = index_c[j]; 
		bootstrap_idx[i].push_back(idx1);
	    }
	}
	
	
    }
    printf("Bootstrap samples: ");
    for(size_t cc=0; cc<m_nclass; cc++)
	printf("class %d = %u, ",cc,bootstrap_idx[cc].size());
    printf("\n");
}

// void PixelDetector::bootstrap_samples(vector< vector<unsigned int> >& input_idx,
// 				      vector< vector<unsigned int> >& bootstrap_idx)
// 
// {
//     bootstrap_idx.clear(); bootstrap_idx.resize(m_nclass);
//     for(size_t i=0; i<m_nclass ; i++){
// 	vector<unsigned int> index_c = input_idx[i];
// 	unsigned int idx_sz = index_c.size();
// 	float idx_sz_quotient1 =  (float)floor(m_class_bias[i]);
// 	unsigned int idx_sz_quotient = (unsigned int)idx_sz_quotient1;
// 	
// 	unsigned int idx_sz_remainder = (unsigned int)(idx_sz*m_class_bias[i]) - (idx_sz*idx_sz_quotient);
// 	
// 	bootstrap_idx[i].clear();
// 	for(size_t qq=0; qq<idx_sz_quotient; qq++){
// 	    for(size_t j=0; j< idx_sz ;j++){
// 		unsigned int idx1 = index_c[j]; 
// 		bootstrap_idx[i].push_back(idx1);
// 	    }
// 	}
// 	for(size_t j=0; j< idx_sz_remainder ;j++){
// 	    int iSecret = (int)(rand() % idx_sz);
// 	    unsigned int idx1 = index_c[iSecret]; 
// 	    bootstrap_idx[i].push_back(idx1);
// 	}
//     }
//     printf("Bootstrap samples: ");
//     for(size_t cc=0; cc<m_nclass; cc++)
// 	printf("class %d = %u, ",cc,bootstrap_idx[cc].size());
//     printf("\n");
// }



// void PixelDetector::bootstrap_samples(vector< vector<unsigned int> >& input_idx,
// 				      vector< vector<unsigned int> >& bootstrap_idx)
// 
// {
//     bootstrap_idx.clear(); bootstrap_idx.resize(m_nclass);
//     for(size_t i=0; i<m_nclass ; i++){
// 	vector<unsigned int> index_c = input_idx[i];
// 	unsigned int idx_sz = index_c.size();
// 	unsigned int idx_sz_bs = (unsigned int) (idx_sz*m_class_bias[i]);
// 	unsigned int idx_sz_extra = idx_sz_bs - idx_sz;
// 	if (idx_sz_extra<0)
// 	    printf("something wrong with class bias\n");
// 	
// 	bootstrap_idx[i].clear();
// 	for(size_t j=0; j< idx_sz ;j++){
// 	    unsigned int idx1 = index_c[j]; 
// 	    bootstrap_idx[i].push_back(idx1);
// 	}
// 	for(size_t j=0; j< idx_sz_extra ;j++){
// 
// 	    int iSecret = (int)(rand() % idx_sz);
// 	    unsigned int idx1 = index_c[iSecret]; 
// 	    bootstrap_idx[i].push_back(idx1);
// 	}
//     }
//     printf("Bootstrap samples: ");
//     for(size_t cc=0; cc<m_nclass; cc++)
// 	printf("class %d = %u, ",cc,bootstrap_idx[cc].size());
//     printf("\n");
// }

void PixelDetector::learn_classifier_multiclass()
{
    vector< vector<unsigned int> > bootstrap_idx;
    bootstrap_samples(m_dis_idx, bootstrap_idx);
    
    vector<int>& ds_labels =  m_dtst_ds.get_labels();  
    vector<unsigned int> trn_idx;
    
	
    vector< vector<float> > cum_train_features;
    vector<int> cum_train_labels ;
    
    size_t ipair=0;
    size_t maxsz = 0, maxid;
    
    vector<unsigned int> new_idx;
    vector<int> new_lbl;
    for(size_t i=0; i < m_nclass; i++){
	new_idx.insert(new_idx.end(), bootstrap_idx[i].begin(), bootstrap_idx[i].end());
	vector<unsigned int>& tmp_idx = bootstrap_idx[i];
	for(size_t j=0; j< tmp_idx.size(); j++){
	    int lbl = ds_labels[tmp_idx[j]];
	    new_lbl.push_back(lbl);
	}
    }
	    
    m_dtst_ds.clear_trn_labels();
    trn_idx.clear();
    
    trn_idx.insert(trn_idx.end(), new_idx.begin(), new_idx.end());
    m_dtst_ds.append_trn_labels(new_lbl);
    m_dtst_ds.get_train_data(trn_idx, cum_train_features, cum_train_labels);
    m_clfr_mult->learn(cum_train_features, cum_train_labels);
    
}

float PixelDetector::measure_info_old(std::vector<float>& gprob, std::vector<float>& dprob)
{
    std::vector<float> pavg(gprob.size(), 0);
    float Hg=0, Hd=0, Havg = 0, KL_dg=0, KL_gd=0;;
    float hmax = log2f(m_nclass);
    
    float eps1=1e-9;
    float gprob_diff=0;
    for(size_t dim=0; dim< gprob.size()-1; dim++){
	gprob_diff = (gprob_diff > fabs(gprob[dim]-gprob[dim+1]) ? gprob_diff: fabs(gprob[dim]-gprob[dim+1]));
    }
    if(gprob_diff<1e-9)
	return 1e9;
    
    float norm2dist=0;
    float dotp = 0;
//     float class_wt[]={1,1,1};
    
    for(size_t i=0; i<gprob.size(); i++){
	float dprob_clipped = (dprob[i]< EPS) ? EPS : dprob[i];
	if (gprob[i] > 0.0){
	    Hg += (-gprob[i]* log2f(gprob[i]));
	    KL_gd += (gprob[i]*log2f((gprob[i]/dprob_clipped)));
	}
	pavg[i] = (gprob[i]+dprob[i])/2;
	
	norm2dist += (gprob[i]-dprob[i])*(gprob[i]-dprob[i]);
    }
    
    for(size_t i=0; i<dprob.size(); i++){
	float gprob_clipped = (gprob[i]<EPS) ? EPS : gprob[i];
	if (dprob[i] > 0.0){
	    Hd += (-dprob[i]* log2f(dprob[i]));
	    KL_dg += (dprob[i]*log2f((dprob[i]/gprob_clipped)));
	}
	
	if(pavg[i]>0.0){
	    Havg += (-pavg[i]*log2f(pavg[i]));
	}
	
	float class_wt = (m_class_bias[i]<EPS)? EPS:m_class_bias[i];
// 	class_wt *= (i==1?2:1);
	class_wt = 1.0/class_wt;
	
	dotp += (gprob[i]*dprob[i]*class_wt);
// 	dotp += (gprob[i]*dprob[i]);
    }
    
    float JS = Havg - 0.5*Hg -0.5*Hd;
    float entropy = (1.0/(2*hmax))*(Hd + Hg);
    float KL_sym = KL_dg + KL_gd;
    
//     float ret = (0.0*entropy) + KL_sym; 
//     float ret = norm2dist;
    
    float ret = dotp;
//     float ret = (0.0*entropy) + JS; 
    
    return ret;
}
float PixelDetector::measure_info_boundary(std::vector<float>& gprob, std::vector<float>& dprob)
{
    float eps1=1e-9;
    float gprob_diff=0;
    for(size_t dim=0; dim< gprob.size()-1; dim++){
	gprob_diff = (gprob_diff > fabs(gprob[dim]-gprob[dim+1]) ? gprob_diff: fabs(gprob[dim]-gprob[dim+1]));
    }
    if(gprob_diff<1e-9)
	return 0;
    
    float dotp = 0;
    
    float max_pred = 0;
    
    float diffp = (gprob[1]-dprob[1])*(gprob[1]-dprob[1]);
    for(size_t dim=0; dim< gprob.size(); dim++){
	
	diffp = fabs(gprob[dim]-dprob[dim]);
// // // // 	diffp = gprob[dim]*(fabs(gprob[dim]-dprob[dim]));
	
	max_pred = (max_pred<diffp) ? diffp : max_pred;
    }
    return (max_pred);
  
}
float PixelDetector::measure_info(std::vector<float>& gprob, std::vector<float>& dprob)
{
    float eps1=1e-9;
    float gprob_diff=0;
    for(size_t dim=0; dim< gprob.size()-1; dim++){
	gprob_diff = (gprob_diff > fabs(gprob[dim]-gprob[dim+1]) ? gprob_diff: fabs(gprob[dim]-gprob[dim+1]));
    }
    if(gprob_diff<1e-9)
	return 1e9;
    
  
    float dotp=0;
    float class_wt;
    for(size_t i=0; i<dprob.size(); i++){
// 	class_wt = 1.0/m_prior_belief[i];
	
// 	class_wt = (m_class_bias[i]<EPS)? EPS:m_class_bias[i];
// 	class_wt = 1.0/class_wt;
	
// 	dotp += (gprob[i]*dprob[i]*class_wt);
	dotp += (gprob[i]*dprob[i]);
    }
    
    
    float ret = dotp;
//     float ret = (0.0*entropy) + JS; 
    
    return ret;
}
float PixelDetector::measure_info_margin(std::vector<float>& gprob, std::vector<float>& dprob)
{
    float eps1=1e-9;
    float gprob_diff=0;
    for(size_t dim=0; dim< gprob.size()-1; dim++){
	gprob_diff = (gprob_diff > fabs(gprob[dim]-gprob[dim+1]) ? gprob_diff: fabs(gprob[dim]-gprob[dim+1]));
    }
//     if(gprob_diff<1e-9)
// 	return 0;
    float gmax=0, dmax=0;
    unsigned int gidx=0, didx=0;
    
    for(size_t dim=0; dim< gprob.size(); dim++){
	
	if (gmax < gprob[dim]){
	    gmax = gprob[dim];
	    gidx = dim;
	}
	
	if (dmax < dprob[dim]){
	    dmax = dprob[dim];
	    didx = dim;
	}
    }
    
    if (gidx!=didx)
      int pp=1;
    
    float g2ndmax=0, d2ndmax=0;
    unsigned int g2ndidx=0, d2ndidx=0;
    
    for(size_t dim=0; dim< gprob.size(); dim++){
	
	if ((g2ndmax < gprob[dim]) && (gidx != dim)){
	    g2ndmax = gprob[dim];
	    g2ndidx = dim;
	}
	
	if ((d2ndmax < dprob[dim]) && (didx != dim)){
	    d2ndmax = dprob[dim];
	    d2ndidx = dim;
	}
    }

    std::vector<float> gmargin(gprob.size(), 0);
    std::vector<float> dmargin(dprob.size(), 0);
    
    if (gidx==1)
	gmargin[gidx] = gmax;
    else
	gmargin[gidx] = gmax - gprob[1];
    if (didx==1)
	dmargin[didx] = dmax;
    else
	dmargin[didx] = dmax - dprob[1];
//     gmargin[gidx] = (gmax - g2ndmax);
//     dmargin[didx] = (dmax - d2ndmax);
    
    float bias=1.0;
    
    if ((gidx==3 && didx!=2) || (didx==3 && gidx!=2))
      bias=0.75;
    
    float diffp = 0;
    for(size_t dim=0; dim< gprob.size(); dim++)
	diffp += bias*fabs(gmargin[dim] - dmargin[dim]);
    
    return diffp;
    
}

void PixelDetector::find_most_informative(std::multimap<float, unsigned int>&  info_list)
{
  
    info_list.clear();
    ClassProbIterator cit = m_class_prob_gen.begin();
    for(; cit!=m_class_prob_gen.end() ; cit++){
	unsigned int idx = cit->first;
	if (m_all_unique_idx.find(idx)!=m_all_unique_idx.end()){ // already in trn set
// 	    printf("already in trn set\n");
	    continue;
	}
	std::vector<float>& gprob = cit->second;
	
	std::vector<float>& dprob = m_class_prob_dis[idx];
	
	float info_val = measure_info_margin(gprob, dprob);
	
	float deg= m_wtmat[0]->get_degree(idx);
	float sqrtdeg = (float)sqrt(deg);
	
// 	float info_val1 = sqrtdeg>0? info_val*sqrtdeg: info_val;
	float info_val1= info_val;
	
	info_list.insert(std::make_pair(info_val1, idx));
    }
}
bool PixelDetector::add_new_points(unsigned int eta)
{
    std::multimap<float, unsigned int >  info_list;
    find_most_informative(info_list);
    
    vector< vector<float> >& ds_features = m_dtst_ds.get_features();
    vector<int>& ds_labels =  m_dtst_ds.get_labels();  
    
    vector<size_t> count_dis(m_nclass,0);
    vector<size_t> count_gen(m_nclass,0);
    vector<unsigned int> nincorrect_gen(m_nclass,0);
    vector<unsigned int> nincorrect_dis(m_nclass,0);
    
    vector<float> disagreements(eta,0);
    
    bool no_more_samples=true;
    
    m_mostrecent_idx.clear();
    m_mostrecent_idx.resize(m_nclass);
    
    std::multimap<float, unsigned int >::reverse_iterator cit = info_list.rbegin();
    for(size_t i=0; i <eta && cit!=info_list.rend(); i++, cit++){
//     std::multimap<float, unsigned int >::iterator cit = info_list.begin();
//     for(size_t i=0; i <eta && cit!=info_list.end(); i++, cit++){
	float info = cit->first;
	unsigned int idx = cit->second;
	disagreements[i] = info;
	if (disagreements[0] < 0.9)
	    int debug_pp=1;
	
	std::vector<float>& gprob = m_class_prob_gen[idx];
	
	std::vector<float>& dprob = m_class_prob_dis[idx];
	
	int label = ds_labels[idx];
	printf("idx: %u, pred diff: %.4f, label: %d\n", m_all_cidx[idx], info, label );
	printf("generative: ");
	for(size_t cc=0; cc<m_nclass; cc++)
	    printf("%.4f  ",gprob[cc]);
	printf("\n");
	printf("discriminative: ");
	for(size_t cc=0; cc<m_nclass; cc++)
	    printf("%.4f  ",dprob[cc]);
	printf("\n");
		
	int label_dis = label_from_prob(m_class_prob_dis[idx]);
	if (label_dis != label){
	    (nincorrect_dis[label])++;
	}
	m_dis_idx[label].push_back(idx);
	m_all_unique_idx.insert(std::make_pair(idx,label));
	(count_dis[label])++;
	no_more_samples=false;
// 	}
	
	int label_gen = label_from_prob(m_class_prob_gen[idx]);
	if (label_gen != label)
	    (nincorrect_gen[label])++;
	m_gen_idx[label].push_back(idx);
// 	m_all_unique_idx.insert(std::make_pair(idx,label));
	(count_gen[label])++;
// 	}
	m_mostrecent_idx[label].push_back(idx);
// 	if (label_dis != label || label_gen != label){
// 	
// 	}
    }
    trnset_size_class();
//     recompute_class_bias();
    printf("class bias: ");
    for(size_t cc=0; cc<m_nclass; cc++)
	printf("%.4f  ",m_class_bias[cc]);
    printf("\n");
    
    printf("Added Discriminative: ");
    for(size_t cc=0; cc<m_nclass; cc++)
	printf("class %d = %u, ",cc,count_dis[cc]);
    printf("\n");
    printf("Added Generative: ");
    for(size_t cc=0; cc<m_nclass; cc++)
	printf("class %d = %u, ",cc,count_gen[cc]);
    printf("\n");
//     printf("Added: class 0: %u, class 1: %u, class 2: %u\n", count_c[0], count_c[1],count_c[2]);
    printf("Discriminant wrong: ");
    for(size_t cc=0; cc<m_nclass; cc++)
	printf("class %d = %u, ",cc,nincorrect_dis[cc]);
    printf("\n");
    int query_incorrect=0;
    printf("Generative wrong: ");
    for(size_t cc=0; cc<m_nclass; cc++){
	printf("class %d = %u, ",cc,nincorrect_gen[cc]);
	query_incorrect += (nincorrect_dis[cc]+nincorrect_gen[cc]);
    }
    printf("\n");
    
    if (disagreements[0] < 0.8)
	m_count_low_disagree++;
    else 
	m_count_low_disagree  = 0;
      
    if (m_count_low_disagree > 4)
	no_more_samples = true;
 
//     no_more_samples=true;
//     if (query_incorrect>0)
//         no_more_samples=false;
    
    return no_more_samples;
}

int PixelDetector::label_from_prob(vector<float>& prob)
{
    int max_loc=0;
    float max_val=prob[max_loc];
    
    for(size_t i=0; i<prob.size(); i++)
	if (prob[i]>max_val){
	    max_val = prob[i];
	    max_loc = i;
	}
    return max_loc;
}






void PixelDetector::compute_confusion(ClassProb& final_prob)
{
  
    vector<int>& ds_labels =  m_dtst_ds.get_labels();  
    std::map<unsigned int, vector<float> >::iterator cit = final_prob.begin();
    
    std::vector< std::vector<float> > confusion_matrix(m_nclass); 
    for(size_t i=0; i< m_nclass; i++)
	confusion_matrix[i].resize(m_nclass,0);
    
    float ntotal=0;
    float ncorrect_zero=0, nfalse_zero=0;
    for(; cit!=final_prob.end(); cit++){
	std::vector<float>& class_probs = (cit->second);
	float maxprob =0;
	unsigned int max_idx;
	for(size_t j=0; j < class_probs.size(); j++ )
	    if (class_probs[j] > maxprob){
		maxprob = class_probs[j];
		max_idx = j;
	    }
	    
	int actual_label = ds_labels[cit->first];

	(confusion_matrix[actual_label][max_idx])++;
	ntotal++;
	if(actual_label==1 && class_probs[1]<0.01)
	    nfalse_zero++;
	
	if(actual_label!=1 && class_probs[1]<0.01)
	    ncorrect_zero++;
	
    }
    float ncorrect=0;
    for(size_t i=0; i<m_nclass; i++){
	ncorrect += 100*confusion_matrix[i][i]/ntotal;
	for(size_t j=0; j< m_nclass; j++){
	    printf("%.3f   ", 100*confusion_matrix[i][j]/ntotal);
	}
	printf("\n");
    }
  
    printf("Test set size = %.1f, correct = %.2f\n",ntotal, ncorrect);
    printf("zero prob: false= %.4f, correct = %.4f\n",nfalse_zero/ntotal, ncorrect_zero/ntotal);
  
}


void PixelDetector::convert2multiclass_prob(std::vector< std::vector <float> >& preds, 
					    std::vector< std::vector <int> >& pair_ids,
					    ClassProb& final_prob)
{
    unsigned int npairs = m_nclass*(m_nclass-1)/2;
  
    float **r = new float*[m_nclass];
    for(size_t i =0; i<m_nclass;i++)
	r[i] = new float[m_nclass];
    float* pc = new float[m_nclass];
    
    bool valid;
    for(size_t j=0; j< preds[0].size(); j++){
	valid=true;
	for (size_t c=0; c<npairs; c++){
	    float pred = preds[c][j];
	    if (pred<0){
		valid= false;
		break;
	    }
	    int class0 = pair_ids[c][0];
	    int class1 = pair_ids[c][1];
	    
	    r[class0][class1] = pred;
	    r[class1][class0] = 1 - pred;
	}
	if (valid){
	    multiclass_probability(m_nclass, r, pc);
	    
	    std::vector<float> clprob(m_nclass);
	    memcpy(clprob.data(), pc, m_nclass*sizeof(float));
	    
	    final_prob.insert(std::make_pair(j, clprob));
	}
    }
    
    delete pc;
    for(size_t i =0; i<m_nclass;i++)
	delete[] r[i];
    delete r;
  
}


// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
void PixelDetector::multiclass_probability(int k, float **r, float *p)
{
	int t,j;
	int iter = 0, max_iter=max(100,k);
	float **Q= new float*[k];//Malloc(float *,k);
	float *Qp= new float[k];//Malloc(float,k);
	float pQp, eps=0.005/k;
	
	for (t=0;t<k;t++)
	{
		p[t]=1.0/k;  // Valid if k = 1
		Q[t]= new float[k];//Malloc(float,k);
		Q[t][t]=0;
		for (j=0;j<t;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=Q[j][t];
		}
		for (j=t+1;j<k;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=-r[j][t]*r[t][j];
		}
	}
	for (iter=0;iter<max_iter;iter++)
	{
		// stopping condition, recalculate QP,pQP for numerical accuracy
		pQp=0;
		for (t=0;t<k;t++)
		{
			Qp[t]=0;
			for (j=0;j<k;j++)
				Qp[t]+=Q[t][j]*p[j];
			pQp+=p[t]*Qp[t];
		}
		float max_error=0;
		for (t=0;t<k;t++)
		{
			float error=fabs(Qp[t]-pQp);
			if (error>max_error)
				max_error=error;
		}
		if (max_error<eps) break;
		
		for (t=0;t<k;t++)
		{
			float diff=(-Qp[t]+pQp)/Q[t][t];
			p[t]+=diff;
			pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/(1+diff)/(1+diff);
			for (j=0;j<k;j++)
			{
				Qp[j]=(Qp[j]+diff*Q[t][j])/(1+diff);
				p[j]/=(1+diff);
			}
		}
	}
	if (iter>=max_iter)
		printf("Exceeds max_iter in multiclass_prob\n");
	for(t=0;t<k;t++) delete Q[t]; //free(Q[t]);
	delete Q; //free(Q);
	delete Qp;//free(Qp);
}

void PixelDetector::build_wtmat(bool read_off, string feature_filename, double w_dist_thd){

//     double w_dist_thd = 2;
    printf("Computing wt matrix with weight thd = %.4lf\n",w_dist_thd);

    unsigned found = feature_filename.find_last_of("//");
    string feature_dir =  feature_filename.substr(0,found);
    string wtmat_filename = feature_dir;
    wtmat_filename += "/";
    wtmat_filename += "weight_matrix";
    unsigned int m_nfeat= m_dtst_ds.get_features().size();
    char tmp[1024];
    sprintf(tmp, "_%d",m_nfeat);
    wtmat_filename += tmp;
    sprintf(tmp, "_%.1lf.txt",w_dist_thd);
    wtmat_filename += tmp;
    
    unsigned int npairs = m_nclass*(m_nclass-1)/2;
    m_wtmat.clear();
    m_wtmat.resize(npairs);
    if(read_off){
	m_wtmat[0] = new WeightMatrix_iter(m_nclass);
	m_wtmat[0]->read_matrix((char*)wtmat_filename.c_str());
// 	m_wtmat[0]->factorize();
    }
    else{
	vector< vector<float> >& ds_features = m_dtst_ds.get_features();
	std::vector<unsigned int> ignore_list;
	m_wtmat[0] = new WeightMatrix_iter(m_nclass, w_dist_thd, ignore_list);
	m_wtmat[0]->weight_matrix_parallel(ds_features, false);
	m_wtmat[0]->factorize();
// // 	m_wtmat[0]->write_matrix((char* )wtmat_filename.c_str());
    }
    printf("nonzero ratio: %.4f\n", 100*m_wtmat[0]->nnz_pct());
//     for(size_t j=1; j<npairs; j++){
// 	m_wtmat[j] = new WeightMatrix_iter;
// 	m_wtmat[j]->read_matrix((char*)wtmat_filename.c_str());
// 	m_wtmat[j]->copy_factor_from(m_wtmat[0]);
// // 	m_wtmat[j]->free_memory();
//     }

//     m_wtmat[0]->free_memory();
}
void PixelDetector::add_extra_samples(size_t nextra)
{
    
    std::vector<unsigned int> idx1;
    vector<int>& ds_labels =  m_dtst_ds.get_labels();

    idx1.clear();
    for(size_t i=0; i<ds_labels.size(); i++){
	if ((ds_labels[i] == 1) && (m_all_unique_idx.find(i) == m_all_unique_idx.end())){
	    idx1.push_back(i);
	}
      
    }
    random_shuffle(idx1.begin(), idx1.end());
    if(idx1.size() > nextra)
	idx1.erase(idx1.begin()+nextra, idx1.end());
    
//     m_dis_idx[1].insert(m_dis_idx[1].end(), idx1.begin(), idx1.end());
    m_gen_idx[1].insert(m_gen_idx[1].end(), idx1.begin(), idx1.end());
    for(size_t i=0; i< idx1.size(); i++)
	m_all_unique_idx.insert(std::make_pair(idx1[i], 1 ));

}

void PixelDetector::get_random_stpoints(int label, unsigned int len, vector<unsigned int>& idx)
{ 
    vector<int>& ds_labels =  m_dtst_ds.get_labels();

    idx.clear();
    for(size_t i=0; i<ds_labels.size(); i++){
	if (ds_labels[i] == label){
	    idx.push_back(i);
	}
      
    }
    random_shuffle(idx.begin(), idx.end());
    if(idx.size() > len)
	idx.erase(idx.begin()+len, idx.end());
    else if (idx.size() < len)
	printf("Initial size smaller than %u for class %d\n",len, label);
}
void PixelDetector::get_random_stpoints2(int label, unsigned int len, vector<unsigned int>& idx)
{ 
    vector<int>& ds_labels =  m_dtst_ds.get_labels();

    idx.clear();
    for(size_t i=0; i<ds_labels.size(); i++){
	if (ds_labels[i] == label && ((m_wtmat[0])->get_degree(i)>0)){
	    idx.push_back(i);
	}
      
    }
    random_shuffle(idx.begin(), idx.end());
    if(idx.size() > len)
	idx.erase(idx.begin()+len, idx.end());
    else if (idx.size() < len)
	printf("Initial size smaller than %u for class %d\n",len, label);
}

void PixelDetector::select_initial_points_kmeans()
{
    vector<int>& ds_labels =  m_dtst_ds.get_labels();
    vector< vector<float> >& ds_features =  m_dtst_ds.get_features();
    
    std::vector< std::vector<float> > scaled_features;
    (m_wtmat[0])->scale_features(ds_features, scaled_features);
    
    vector<unsigned int> init_trn_idx;
    
    kMeans km((m_nclass*m_init_sz), 100, 1e-2);
    km.compute_centers(scaled_features, init_trn_idx);
    
    m_all_unique_idx.clear();
    m_gen_idx.clear();
    m_gen_idx.resize(m_nclass);
    m_dis_idx.clear();
    m_dis_idx.resize(m_nclass);

    unsigned int trn_sza = ((m_nclass*m_init_sz)>init_trn_idx.size()? init_trn_idx.size():(m_nclass*m_init_sz) );
    for (unsigned int i=0; i < trn_sza; i++){
	unsigned int index1 = init_trn_idx[i];
	int label = ds_labels[index1];
	m_gen_idx[label].push_back(index1);
	m_dis_idx[label].push_back(index1);
	
	m_all_unique_idx.insert(std::make_pair( index1, label ));
    }
    
    recompute_class_bias();
    bootstrap_samples(m_gen_idx, m_dis_idx);
    for (int i=0; i < m_nclass; i++)
	m_class_bias[i] = 1.0;
}
void PixelDetector::select_initial_points_biased()
{
  
    vector<int>& ds_labels =  m_dtst_ds.get_labels();
    m_all_unique_idx.clear();
    vector<unsigned int> index_c;
    for (int i=0; i < m_nclass; i++){
	get_random_stpoints2( i,m_init_sz, index_c);
	m_gen_idx.push_back(index_c);
	m_dis_idx.push_back(index_c);
	for(size_t j=0; j< index_c.size();j++){
	    m_all_unique_idx.insert(std::make_pair(m_gen_idx[i][j], ds_labels[m_gen_idx[i][j]] ));
	    
	}
	
    }
    recompute_class_bias3(); 
    // if #mito boundary labels > #mito labels   
    size_t allowed_sz = m_dis_idx[2].size()/2;
    if( (m_nclass>3) && (m_dis_idx.size()>3 )){ 
	if(m_dis_idx[3].size()> allowed_sz){
	// if #mito boundary labels > #mito labels
	    printf("discarding mito border samples\n");
	    size_t nmito_labels = allowed_sz;
	    m_dis_idx[3].erase(m_dis_idx[3].begin()+nmito_labels, m_dis_idx[3].end());
	}
    }

//     bootstrap_samples(m_gen_idx, m_dis_idx);
    
//     get_random_stpoints2( 1, m_init_sz/2, index_c);
//     m_dis_idx[1].insert(m_dis_idx[1].end(), index_c.begin(), index_c.end());
	
    for (int i=0; i < m_nclass; i++)
	m_class_bias[i] = 1.0;
    
    m_mostrecent_idx.clear();
    m_mostrecent_idx.resize(m_nclass);
    for (int i=0; i < m_nclass; i++)
	m_mostrecent_idx[i]=m_gen_idx[i];
    
}
void PixelDetector::select_initial_points_fixed()
{
  
    vector<size_t> init_max_sizes(m_nclass);
    init_max_sizes[0]= m_init_sz; 
    init_max_sizes[1]= m_init_sz;
    init_max_sizes[2]= m_init_sz/3;
    init_max_sizes[3]= m_init_sz/3;
    
  
    vector<int>& ds_labels =  m_dtst_ds.get_labels();
    m_all_unique_idx.clear();
    vector<unsigned int> index_c;
    vector<unsigned int> index_c2;
    for (int i=0; i < m_nclass; i++){
	get_random_stpoints2( i,m_init_sz, index_c);
	m_gen_idx.push_back(index_c);
	for(size_t j=0; j< index_c.size();j++){
	    m_all_unique_idx.insert(std::make_pair(m_gen_idx[i][j], ds_labels[m_gen_idx[i][j]] ));
	    
	}
	index_c2.clear();
	index_c2.assign(index_c.begin(), index_c.begin()+init_max_sizes[i]);
	m_dis_idx.push_back(index_c2);
	
    }
    for (int i=0; i < m_nclass; i++)
	m_class_bias[i] = 1.0;
    
    m_mostrecent_idx.clear();
    m_mostrecent_idx.resize(m_nclass);
    for (int i=0; i < m_nclass; i++)
	m_mostrecent_idx[i]=m_gen_idx[i];

}
void PixelDetector::select_initial_points()
{
  
    vector<int>& ds_labels =  m_dtst_ds.get_labels();
    
    m_mostrecent_idx.clear();
    m_gen_idx.clear();
    m_dis_idx.clear();
    m_all_unique_idx.clear();
    for (int i=0; i < m_nclass; i++){
	vector<unsigned int> index_c;
// 	if (i==3)
// 	    get_random_stpoints( i,(m_init_sz), index_c);
// 	else
	    get_random_stpoints( i,m_init_sz, index_c);
	m_gen_idx.push_back(index_c);
	m_dis_idx.push_back(index_c);
	m_mostrecent_idx.push_back(index_c);
	for(size_t j=0; j< index_c.size();j++)
	    m_all_unique_idx.insert(std::make_pair(index_c[j], ds_labels[index_c[j]] ));
    }
}

void PixelDetector::select_initial_points2()
{
    vector<unsigned int> rand_idx;
    vector<int>& ds_labels =  m_dtst_ds.get_labels();
    rand_idx.clear();
    for(unsigned int i=0; i<ds_labels.size(); i++)
	rand_idx.push_back(i);
    
    size_t len = m_init_sz*m_nclass;
    random_shuffle(rand_idx.begin(), rand_idx.end());
    if(rand_idx.size() > len)
	rand_idx.erase(rand_idx.begin()+len, rand_idx.end());
    else if (rand_idx.size() < len)
	printf("Initial size smaller than %u\n",len);
    
    m_all_unique_idx.clear();
    m_gen_idx.clear();
    m_gen_idx.resize(m_nclass);
    for(size_t j=0; j< rand_idx.size();j++){
	unsigned int idx = rand_idx[j];
	int label = ds_labels[idx];
	m_gen_idx[label].push_back(idx);
	m_all_unique_idx.insert(std::make_pair(idx, label ));
    }
    
    
    m_mostrecent_idx.clear();
    m_mostrecent_idx.resize(m_nclass);
    m_dis_idx.clear();
    m_dis_idx.resize(m_nclass);
    for (int i=0; i < m_nclass; i++){
	m_mostrecent_idx[i]=m_gen_idx[i];
	m_dis_idx[i]=m_gen_idx[i];
    }
    
    for(int i=0; i<m_nclass; i++)
      printf("class %d = %u, ",i, m_gen_idx[i].size()); 
    printf("\n");
}

void PixelDetector::save_classifier(string clfr_name)
{
    unsigned int npairs = m_nclass*(m_nclass-1)/2;
    char tmp[256];
    for(size_t i=0; i<npairs; i++){
	string clfr_name1 = clfr_name.substr(0, clfr_name.length()-3);
	sprintf(tmp,"%u.h5",i);
	clfr_name1 +="_";
	clfr_name1 += tmp;
	
	m_pclfrs[i]->save_classifier(clfr_name1.c_str());
    }
}

void PixelDetector::save_classifier_m(string clfr_name)
{
    m_clfr_mult->save_classifier(clfr_name.c_str());
}
void PixelDetector::save_classifier_inter(string clfr_name)
{
    unsigned int npairs = m_nclass*(m_nclass-1)/2;
    char tmp[256];
    for(size_t i=0; i<npairs; i++){
	string clfr_name1 = clfr_name.substr(0, clfr_name.length()-3);
	sprintf(tmp,"%u.h5",i);
	clfr_name1 +="_";
	clfr_name1 += tmp;
	
	sprintf(tmp,"%u.h5",trnset_size());
	clfr_name1 +="_";
	clfr_name1 += tmp;
	
	m_pclfrs[i]->save_classifier(clfr_name1.c_str());
    }
}
void PixelDetector::save_classifier_inter_m(string clfr_name)
{
    char tmp[256];
    string clfr_name1 = clfr_name.substr(0, clfr_name.length()-3);
    
    sprintf(tmp,"%u.h5",trnset_size());
    clfr_name1 +="_";
    clfr_name1 += tmp;
    
    m_clfr_mult->save_classifier(clfr_name1.c_str());
}
void PixelDetector::load_classifier(string clfr_name)
{
    unsigned int npairs = m_nclass*(m_nclass-1)/2;
    char tmp[256];
    for(size_t i=0; i<npairs; i++){
	string clfr_name1 = clfr_name.substr(0, clfr_name.length()-3);
	sprintf(tmp,"%u.h5",i);
	clfr_name1 +="_";
	clfr_name1 += tmp;
	
	m_pclfrs[i]->load_classifier(clfr_name1.c_str());
    }
}

void PixelDetector::load_classifier_m(string clfr_name)
{
  m_clfr_mult->load_classifier(clfr_name.c_str());
}

void PixelDetector::recompute_class_bias()
{
    vector<float> tmp_bias(m_nclass,0);
    float weight, degree;
    vector< vector<float> > conn(m_nclass);
    
    for (int i=0; i < m_nclass; i++){
	float class_cluster_volume=0;
	conn[i].resize(m_nclass, 0.0);
	for(size_t j=0; j< m_gen_idx[i].size();j++){
	    m_wtmat[0]->connection_unlabeled(m_gen_idx[i][j], i, m_all_unique_idx, degree, conn[i]);
	    if (degree>0){  
// 		(tmp_bias[i]) += weight;
		class_cluster_volume += degree;
	    }
// 	    float weight = m_wtmat[0]->get_degree(m_gen_idx[i][j]);
// 	    if (weight>0) weight = 1./sqrt(weight);
	}
	(tmp_bias[i]) /= class_cluster_volume;
	
// 	if (tmp_bias[i]<min_bias){
// 	    min_bias = tmp_bias[i];
// 	    min_class = i;
// 	}
    }
    float **r=new float*[m_nclass];
    for (int i=0; i < m_nclass; i++){
	r[i] = new float[m_nclass];
	float class_sum =0;
	for(int j=0;j<m_nclass; j++)
	    class_sum += conn[i][j];
	for(int j=0;j<m_nclass; j++){
	    r[i][j] = conn[i][j]/class_sum;
	    conn[i][j] /=class_sum;
	}
	r[i][i]=0;
    }   
    
// //     multiclass_probability(m_nclass,r, m_class_bias.data());
    for (int i=0; i < m_nclass; i++){
	m_class_bias[i] = 0;
	for(int j=0;j<m_nclass; j++)
	    if(i!=j)
		(m_class_bias[i]) += r[i][j];
    }
    
    printf("connectivity:\n");
    for (int i=0; i < m_nclass; i++){
	for(int j=0;j<m_nclass; j++){
	    printf("%.3f ",conn[i][j]);
	}
	printf("\n");
    }
    float max_bias=0; int max_class=0;
    printf("class bias:");
    for(int i=0;i<m_nclass; i++){
	printf("%.3f ",m_class_bias[i]);
	if (m_class_bias[i]>max_bias){
	    max_bias = m_class_bias[i];
	    max_class = i;
	}
    }
    printf("\n");
    
    for (int i=0; i < m_nclass; i++){
	m_class_bias[i] /= max_bias;
	m_class_bias[i] *= m_prior_belief[i];
	printf("%.3f ",m_class_bias[i]);
	
	delete[] r[i];
    }    
//     float min_bias=1e12; int min_class=0;
//     printf("class bias:");
//     for(int i=0;i<m_nclass; i++){
// 	printf("%.3f ",m_class_bias[i]);
// 	if (m_class_bias[i]<min_bias){
// 	    min_bias = m_class_bias[i];
// 	    min_class = i;
// 	}
//     }
//     printf("\n");
//     
//     for (int i=0; i < m_nclass; i++){
// 	m_class_bias[i] /= min_bias;
// 	m_class_bias[i] *= m_prior_belief[i];
// 	printf("%.3f ",m_class_bias[i]);
// 	
// 	delete[] r[i];
//     }    
    printf("\n");
    delete [] r;
//     m_class_bias[0] = 8.5; m_class_bias[1] = 3.4; m_class_bias[2]=1; m_class_bias[3]=5.0;
}

void PixelDetector::recompute_class_bias2()
{
    vector<float> tmp_bias(m_nclass,0);
    float weight, degree;
    vector< vector<float> > conn(m_nclass);
    vector< vector<float> > prev_conn(m_nclass);
    vector<float> diff_conn(m_nclass,0);
    
    float boundary_cluster_volume=0, boundary_remain_prob=0;
    conn[1].resize(m_nclass, 0.0);
// // //     m_dis_idx[1] = m_gen_idx[1];
    prev_conn[1].resize(m_nclass, 0.0);
    float cut=0;
    m_dis_idx[1].clear();
    for(size_t j=0; j< m_gen_idx[1].size();j++){
	m_wtmat[0]->connection_unlabeled(m_gen_idx[1][j], 1, m_all_unique_idx, degree, conn[1]);
// // 	if(degree>0){
	    m_dis_idx[1].push_back(m_gen_idx[1][j]);
	    boundary_cluster_volume += degree;
	    for(int cc=0;cc<m_nclass;cc++){
		diff_conn[cc] = conn[1][cc] - prev_conn[1][cc];
		if (cc == 1)
		  cut += diff_conn[cc];
	    }
	    prev_conn[1]=conn[1];
// // 	}
    }
    boundary_remain_prob = cut/boundary_cluster_volume;
    printf("boundary class volume: %f\n", boundary_cluster_volume);
    printf("boundary trans prob: %f\n", boundary_remain_prob);
    
    
    for (int i=0; i < m_nclass; i++){
	if (i==1)
	  continue;
	float class_cluster_volume=0;
	conn[i].resize(m_nclass, 0.0);
	prev_conn[i].resize(m_nclass, 0.0);
	std::multimap<float, unsigned int> rank_by_cut;
	std::map<unsigned int, float> degree_list;
	vector<unsigned int> zero_degree_list;
	for(size_t j=0; j< m_gen_idx[i].size();j++){
	    m_wtmat[0]->connection_unlabeled(m_gen_idx[i][j], i, m_all_unique_idx, degree, conn[i]);
	    if (degree>0){  
		class_cluster_volume += degree;
	    
		cut=0;
		for(int cc=0;cc<m_nclass;cc++){
		    diff_conn[cc] = conn[i][cc] - prev_conn[i][cc];
		    if (cc!=i)
		      cut += diff_conn[cc];
		}
		rank_by_cut.insert(std::make_pair(cut, m_gen_idx[i][j]));
		degree_list.insert(std::make_pair(m_gen_idx[i][j], degree));
		
		prev_conn[i]=conn[i];
	    }
	}
	m_dis_idx[i].clear();
	unsigned int count=0;
	float acc_cluster_volume=0, acc_trans_prob=0;
	std::multimap<float, unsigned int>::reverse_iterator rit = rank_by_cut.rbegin();
	cut=0;
	while ((rit!=rank_by_cut.rend())&&(acc_cluster_volume<boundary_cluster_volume)){
// 	while ((rit!=rank_by_cut.rend())&&(acc_trans_prob<boundary_remain_prob)){
	    unsigned int idx1 = rit->second;
	    cut += rit->first;
	    
	    m_dis_idx[i].push_back(idx1);
	    acc_cluster_volume += (degree_list[idx1]);
	    acc_trans_prob = cut/acc_cluster_volume;
	    rit++;
	}
	printf("class %d volume: %f, ranked list length %u\n",i, acc_cluster_volume, rank_by_cut.size());
	printf("class %d trans prob: %f, ranked list length %u\n",i, acc_trans_prob, rank_by_cut.size());

    }
    float **r=new float*[m_nclass];
    for (int i=0; i < m_nclass; i++){
	r[i] = new float[m_nclass];
	float class_sum =0;
	for(int j=0;j<m_nclass; j++)
	    class_sum += conn[i][j];
	for(int j=0;j<m_nclass; j++){
	    r[i][j] = conn[i][j]/class_sum;
	    conn[i][j] /=class_sum;
	}
	r[i][i]=0;
    }   
    
// //     multiclass_probability(m_nclass,r, m_class_bias.data());
    for (int i=0; i < m_nclass; i++){
	m_class_bias[i] = 0;
	for(int j=0;j<m_nclass; j++)
	    if(i!=j)
		(m_class_bias[i]) += r[i][j];
    }
    
    printf("connectivity:\n");
    for (int i=0; i < m_nclass; i++){
	for(int j=0;j<m_nclass; j++){
	    printf("%.3f ",conn[i][j]);
	}
	printf("\n");
    }
    float max_bias=0; int max_class=0;
    printf("class bias:");
    for(int i=0;i<m_nclass; i++){
	printf("%.3f ",m_class_bias[i]);
	if (m_class_bias[i]>max_bias){
	    max_bias = m_class_bias[i];
	    max_class = i;
	}
    }
    printf("\n");
//     
//     for (int i=0; i < m_nclass; i++){
// 	m_class_bias[i] /= max_bias;
// 	m_class_bias[i] *= m_prior_belief[i];
// 	printf("%.3f ",m_class_bias[i]);
// 	
// 	delete[] r[i];
//     }    
//     float min_bias=1e12; int min_class=0;
//     printf("class bias:");
//     for(int i=0;i<m_nclass; i++){
// 	printf("%.3f ",m_class_bias[i]);
// 	if (m_class_bias[i]<min_bias){
// 	    min_bias = m_class_bias[i];
// 	    min_class = i;
// 	}
//     }
//     printf("\n");
//     
//     for (int i=0; i < m_nclass; i++){
// 	m_class_bias[i] /= min_bias;
// 	m_class_bias[i] *= m_prior_belief[i];
// 	printf("%.3f ",m_class_bias[i]);
// 	
// 	delete[] r[i];
//     }    
//     printf("\n");
//     delete [] r;
//     m_class_bias[0] = 8.5; m_class_bias[1] = 3.4; m_class_bias[2]=1; m_class_bias[3]=5.0;
}


void PixelDetector::recompute_class_bias3()
{
    vector<float> tmp_bias(m_nclass,0);
    float weight, degree;
    vector< vector<float> > conn(m_nclass);
    vector< vector<float> > prev_conn(m_nclass);
    vector<float> diff_conn(m_nclass,0);
    
    float boundary_cluster_volume=0, boundary_remain_prob=0;
    conn[1].resize(m_nclass, 0.0);
// // //     m_dis_idx[1] = m_gen_idx[1];
    prev_conn[1].resize(m_nclass, 0.0);
    float cut=0;
    m_dis_idx[1].clear();
    for(size_t j=0; j< m_gen_idx[1].size();j++){
	m_wtmat[0]->connection_unlabeled(m_gen_idx[1][j], 1, m_all_unique_idx, degree, conn[1]);
// // 	if(degree>0){
	    m_dis_idx[1].push_back(m_gen_idx[1][j]);
	    boundary_cluster_volume += degree;
	    for(int cc=0;cc<m_nclass;cc++){
		diff_conn[cc] = conn[1][cc] - prev_conn[1][cc];
		if (cc == 1)
		  cut += diff_conn[cc];
	    }
	    prev_conn[1]=conn[1];
// // 	}
    }
    boundary_remain_prob = cut/boundary_cluster_volume;
    printf("boundary class volume: %f\n", boundary_cluster_volume);
    printf("boundary trans prob: %f\n", boundary_remain_prob);
    
    
    for (int i=0; i < m_nclass; i++){
	if (i==1)
	  continue;
	float class_cluster_volume=0;
	conn[i].resize(m_nclass, 0.0);
	prev_conn[i].resize(m_nclass, 0.0);
	std::multimap<float, unsigned int> rank_by_cut;
	std::map<unsigned int, float> degree_list;
	vector<unsigned int> zero_degree_list;
	for(size_t j=0; j< m_gen_idx[i].size();j++){
	    m_wtmat[0]->connection_unlabeled(m_gen_idx[i][j], i, m_all_unique_idx, degree, conn[i]);
	    if (degree>0){  
		class_cluster_volume += degree;
	    
		cut=0;
		float cut_other=0;
		for(int cc=0;cc<m_nclass;cc++){
		    diff_conn[cc] = conn[i][cc] - prev_conn[i][cc];
		    if (cc==1)
			cut += diff_conn[cc];
		}
		rank_by_cut.insert(std::make_pair(cut, m_gen_idx[i][j]));
		degree_list.insert(std::make_pair(m_gen_idx[i][j], degree));
		
		prev_conn[i]=conn[i];
	    }
// 	    else
// 		zero_degree_list.push_back(m_gen_idx[i][j]);
	}
	printf("class %d ranked list length %u\n",i, rank_by_cut.size());
	
	m_dis_idx[i].clear();
	unsigned int count=0;
	float acc_cluster_volume=0, acc_trans_prob=0;
	cut=0;

	std::set<unsigned int> already_inserted;
	std::multimap<float, unsigned int>::reverse_iterator rit = rank_by_cut.rbegin();
	while ((rit!=rank_by_cut.rend())&&(acc_cluster_volume<boundary_cluster_volume)){
// 	while ((rit!=rank_by_cut.rend())&&(acc_trans_prob< (cut/boundary_cluster_volume))){
// 	while ((rit!=rank_by_cut.rend())&&(acc_trans_prob<boundary_remain_prob)){
// 	while (rit!=rank_by_cut.rend()){
	    unsigned int idx1 = rit->second;
	    cut += rit->first;
	    
	    m_dis_idx[i].push_back(idx1); 
	    /** remove for ISBI, Davi **/already_inserted.insert(idx1);
	    acc_cluster_volume += (degree_list[idx1]);
	    acc_trans_prob = (cut/acc_cluster_volume);
	    rit++;
	}
	printf("class %d volume: %f\n",i, acc_cluster_volume);
	printf("class %d trans prob: %f\n",i, acc_trans_prob);
	
	std::multimap<float, unsigned int>::iterator fit = rank_by_cut.begin();
// // // // 	while ( fit!=rank_by_cut.end() ){
// // // // 	    unsigned int idx1 = fit->second;
// // // // 	    float cut1 = fit->first;   
// // // // 
// // // // 	    if(cut1>0)
// // // // 	      break;
// // // // 	    if (already_inserted.find(idx1) == already_inserted.end()){
// // // // 		m_dis_idx[i].push_back(idx1);
// // // // 		acc_cluster_volume += (degree_list[idx1]);
// // // // 		cut += cut1;
// // // // 		acc_trans_prob = cut/acc_cluster_volume;
// // // // 	    }
// // // // 	    fit++;
// // // // 	}
	printf("class %d volume: %f\n",i, acc_cluster_volume);
	printf("class %d trans prob: %f\n",i, acc_trans_prob);
    }
    float **r=new float*[m_nclass];
    for (int i=0; i < m_nclass; i++){
	r[i] = new float[m_nclass];
	float class_sum =0;
	for(int j=0;j<m_nclass; j++)
	    class_sum += conn[i][j];
	for(int j=0;j<m_nclass; j++){
	    r[i][j] = conn[i][j]/class_sum;
	    conn[i][j] /=class_sum;
	}
	r[i][i]=0;
    }   
    
// //     multiclass_probability(m_nclass,r, m_class_bias.data());
    for (int i=0; i < m_nclass; i++){
	m_class_bias[i] = 0;
	for(int j=0;j<m_nclass; j++)
	    if(i!=j)
		(m_class_bias[i]) += r[i][j];
    }
    
    printf("connectivity:\n");
    for (int i=0; i < m_nclass; i++){
	for(int j=0;j<m_nclass; j++){
	    printf("%.3f ",conn[i][j]);
	}
	printf("\n");
    }
    float max_bias=0; int max_class=0;
    printf("class bias:");
    for(int i=0;i<m_nclass; i++){
	printf("%.3f ",m_class_bias[i]);
	if (m_class_bias[i]>max_bias){
	    max_bias = m_class_bias[i];
	    max_class = i;
	}
    }
    printf("\n");
}




