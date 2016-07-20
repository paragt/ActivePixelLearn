#ifndef _vigra_watershed_uint
#define _vigra_watershed_uint

#include <map>
#include <set>
//#include <vigra/seededregiongrowing3d.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/labelvolume.hxx>
#include <vigra/multi_watersheds.hxx>

using namespace vigra;
 

typedef vigra::MultiArray<3,unsigned int> UIVolume;
typedef vigra::MultiArray<3,unsigned char> UCVolume;
typedef vigra::MultiArray<3,float> DVolume;


class VigraWatershed{
    size_t _depth; 	
    size_t _width; 	
    size_t _height; 
    

    UIVolume m_ccvol;
    UCVolume m_seedvol;
    UCVolume m_predvol;
    
public:
    VigraWatershed(size_t depth, size_t height, size_t width): _depth(depth), _height(height), _width(width) {
	m_ccvol.reshape(UIVolume::difference_type(_depth, _height, _width));
	m_seedvol.reshape(UCVolume::difference_type(_depth, _height, _width));
	m_predvol.reshape(UCVolume::difference_type(_depth, _height, _width));
    };  	

    void get_volume_ptr(unsigned char** pvolp, unsigned char** svolp){
	*pvolp = m_predvol.data();
	*svolp = m_seedvol.data();
    };
    unsigned int run_watershed(unsigned char* data_vol, unsigned int* lbl_vol, size_t min_cc_sz, size_t min_region_sz){ 	
	
    	unsigned long volsz = _depth*_height*_width;
	unsigned long plane_size = _height*_width;
	unsigned char pred_i;
	
	
	unsigned int max_region_label = labelVolumeWithBackground(m_seedvol, m_ccvol, NeighborCode3DSix(), 0);
	
	unsigned int nwatershed_regions = remove_small_regions(min_cc_sz);
	
	watershedsMultiArray(m_predvol, m_ccvol, DirectNeighborhood, WatershedOptions().turboAlgorithm());
	
	nwatershed_regions = remove_small_regions(min_region_sz);
	
	watershedsMultiArray(m_predvol, m_ccvol, DirectNeighborhood, WatershedOptions().turboAlgorithm());
	
	memcpy((void *)lbl_vol,(void *) (m_ccvol.data()), volsz*sizeof(unsigned int));
	return nwatershed_regions;
    }	

  
    unsigned int remove_small_regions(size_t min_cc_sz){
	unsigned int* ccvolp = m_ccvol.data();
    	unsigned long volsz = _depth*_height*_width;
// 	unsigned char* predvolp = m_predvol.data();
// 	unsigned char* seedvolp = m_seedvol.data();

	std::map<unsigned int, unsigned int> ccsz;
	std::map<unsigned int, unsigned int>::iterator cit;
	for(unsigned long l1d=0; l1d < volsz; l1d++){
	    unsigned int ccid = ccvolp[l1d];
	    if (ccid==0) continue;
	    cit = ccsz.find(ccid);
	    if (cit!=ccsz.end()){
		unsigned int ccsz1 = cit->second;
		if (ccsz1 <= min_cc_sz)
		    (ccsz[ccid])++;
	    }
	    else{
		ccsz.insert(std::make_pair(ccid,1));
	    }
	}	
	unsigned int nwatershed_regions=0;
	std::set<unsigned int> disregard;
	for(cit=ccsz.begin(); cit!=ccsz.end(); cit++ )
	    if (cit->second < min_cc_sz)
		disregard.insert(cit->first);
	    else nwatershed_regions++;
	    
	for(unsigned long l1d=0; l1d < volsz; l1d++){
	    unsigned int ccid = ccvolp[l1d];
	    if (ccid==0) continue;
	    if (disregard.find(ccid) != disregard.end())
		ccvolp[l1d] = 0;
	}	
	return nwatershed_regions;
    }
    
};

#endif
