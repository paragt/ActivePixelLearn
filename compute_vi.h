#ifndef _COMPUTE_VI
#define _COMPUTE_VI

#define Label unsigned int

class LabelCount{
public:
      Label lbl;
      size_t count;
      LabelCount(): lbl(0), count(0) {};	 	 		
      LabelCount(Label plbl, size_t pcount): lbl(plbl), count(pcount) {};	 	 		
};


class ComputeVI{
  
    Label _depth;
    Label _height;
    Label _width;
    
    Label* _vol1;
    Label* _volg;
    
    std::multimap<Label, std::vector<LabelCount> > contingency;	

public:  
  
    ComputeVI(Label pdepth, Label pheight, Label pwidth): _depth(pdepth), _height(pheight), _width(pwidth){};
  
    Label find_max(Label* data, const size_t* dims){
	Label max=0;  	
	size_t plane_size = dims[1]*dims[2];
	size_t width = dims[2];	 	
	for(size_t i=0;i<dims[0];i++){
	    for (size_t j=0;j<dims[1];j++){
		for (size_t k=0;k<dims[2];k++){
		    size_t curr_spot = i*plane_size + j * width + k;
		    if (max<data[curr_spot])
			max = data[curr_spot];	
		}
	    }
	}	
	return max;	
    }

    void compute_contingency_table(){

// 	if(!gtruth)
// 	    return;		

	size_t i,j,k;
// 	Label *vol1 = get_label_volume(); 		
	const size_t dimn[]={_depth, _height, _width};
	    
	Label ws_max = find_max(_vol1, dimn);  	

	contingency.clear();	
	//contingency.resize(ws_max+1);
	unsigned long ws_size = dimn[0]*dimn[1]*dimn[2];	
	for(unsigned long itr=0; itr<ws_size ; itr++){
	    Label wlabel = _vol1[itr];
	    Label glabel = _volg[itr];
	    multimap<Label, vector<LabelCount> >::iterator mit;
	    mit = contingency.find(wlabel);
	    if (mit != contingency.end()){
		vector<LabelCount>& gt_vec = mit->second;
		for (j=0; j< gt_vec.size();j++)
		    if (gt_vec[j].lbl == glabel){
			(gt_vec[j].count)++;
			break;
		    }
		if (j==gt_vec.size()){
		    LabelCount lc(glabel,1);
		    gt_vec.push_back(lc);	
		}
	    }
	    else{
		vector<LabelCount> gt_vec;	
		gt_vec.push_back(LabelCount(glabel,1));	
		contingency.insert(make_pair(wlabel, gt_vec));	
	    }
	}		

// 	delete[] _vol1;	
    }
    
    double compute_vi(Label* pvol1, Label* pvolg){

	_vol1 = pvol1;
	_volg = pvolg;
// 	if(!gtruth)
// 	    return;		

	int j, k;

	compute_contingency_table();

	double sum_all=0;

	int nn = contingency.size();
	    
	multimap<Label, double>  wp;	 		
	multimap<Label, double>  gp;	 		

	for(multimap<Label, vector<LabelCount> >::iterator mit = contingency.begin(); mit != contingency.end(); ++mit){
	    Label i = mit->first;
	    vector<LabelCount>& gt_vec = mit->second; 	
	    wp.insert(make_pair(i,0.0));
	    for (j=0; j< gt_vec.size();j++){
		unsigned int count = gt_vec[j].count;
		Label gtlabel = gt_vec[j].lbl;
	    
		(wp.find(i)->second) += count;	

		if(gp.find(gtlabel) != gp.end())
		    (gp.find(gtlabel)->second) += count;
		else
		    gp.insert(make_pair(gtlabel,count));	

		sum_all += count;
	    }
	    int tt=1;
	}

	double HgivenW=0;
	double HgivenG=0;
				    
	for(multimap<Label, vector<LabelCount> >::iterator mit = contingency.begin(); mit != contingency.end(); ++mit){
	    Label i = mit->first;
	    vector<LabelCount>& gt_vec = mit->second; 	
	    for (j=0; j< gt_vec.size();j++){
		unsigned int count = gt_vec[j].count;
		Label gtlabel = gt_vec[j].lbl;
		
		double p_wg = count/sum_all;
		double p_w = wp.find(i)->second/sum_all; 	
		
		HgivenW += p_wg* log(p_w/p_wg);

		double p_g = gp.find(gtlabel)->second/sum_all;	

		HgivenG += p_wg * log(p_g/p_wg);	

	    }
	}

	printf("VI= %.5f , MergeSplit: ( %.5f , %.5f )\n",HgivenW+HgivenG,HgivenW, HgivenG); 		
	return ((HgivenW+HgivenG));
    }
};

#endif