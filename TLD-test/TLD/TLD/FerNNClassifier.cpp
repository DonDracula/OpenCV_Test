/*
	FerNNClassifier.cpp
	to brief the NN classifer function
	2016.2.10

*/

#include "FerNNClassifier.h"

using namespace cv;
using namespace std;

void FerNNClassifier::read(const FileNode& file){
	//classifier parameters
	valid = (float) file["valid"];
	ncc_thesame = (float)file["ncc_thesame"];
	nstructs = (int)file["num_trees"];
	structSize = (int)file["num_features"];
	thr_fern =(float)file["thr_fern"];
	thr_nn = (float)file["thr_nn"];
	thr_nn_valid = (float)file["thr_nn_valid"];
}

void FerNNClassifier::prepare(const vector<Size>& scales){
	acum = 0;
	//initialize test locations for features
	int totalFeatures = nstructs*structSize;
	features = vector<vector<Feature>>(scales.size(),vector<Feature>(totalFeatures));
		
	RNG& rng = theRNG();
	float x1f,x2f,y1f,y2f;
	int x1, x2, y1, y2;
	for(int i= 0; i<totalFeatures; i++)
	{
		x1f=(float) rng;
		y1f = (float)rng;
		x2f = (float)rng;
		y2f = (float)rng;
		for(int s = 0; s<scales.size(); s++){
			x1 = x1f * scales[s].width;
			y1 = y1f * scales[s].height;
			x2 = x2f * scales[s].height;
			y2 = y2f * scales[s].height;
			features[s][i] = Feature(x1, y1, x2, y2);
		}
	}

	//Thresholds
	thrN = 0.5*nstructs;

	//Initialize Posteriors
	for(int i = 0; i < nstructs; i++){
		posteriors.push_back(vector<float>(pow(2.0,structSize),0));
		pCounter.push_back(vector<int>(pow(2.0,structSize),0));
		nCounter.push_back(vector<int>(pow(2.0f,structSize),0));
	}
}

void FerNNClassifier::getFeatures(const cv::Mat& image, const int& scale_idx, vector<int>& fern){
	int leaf;
	for(int t=0; t<nstructs; t++){
		leaf = 0;
		for(int f=0; f< structSize; f++){
			leaf = (leaf <<1) + features[scale_idx][t*nstructs+f](image);
		}
		fern[t] = leaf;
	}
}

float FerNNClassifier::measure_forest(vector<int> fern) {
	float votes = 0;
	for(int i = 0; i < nstructs; i++){
		votes += posteriors[i][fern[i]];
	}
	return votes;
}

void FerNNClassifier::update(const vector<int>& fern, int C, int N){
	int idx;
	for( int i = 0; i< nstructs; i++){
		idx = fern[i];
		(C==1) ? pCounter[i][idx] += N : nCounter[i][idx] += N;
		if(pCounter[i][idx]==0){
			posteriors[i][idx] = 0;
		}
		else{
			posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);
		}
	}
}

void FerNNClassifier::trainF(const vector<std::pair<vector<int>,int>>& ferns, int resample){

	thrP = thr_fern*nstructs;
	for(int i = 0; i< ferns.size(); i++){
		if(ferns[i].second==1){
			if(measure_forest(ferns[i].first)<=thrP)
				update(ferns[i].first,1,1);
		}
		else{
			if(measure_forest(ferns[i].first)>= thrN)
				update(ferns[i].first,0,1);
		}
	}
}

void FerNNClassifier:: trainNN(const vector<cv::Mat>& nn_examples){
	float conf, dummy;
	vector<int> y(nn_examples.size(),0);
	y[0] = 1;
	vector<int> isin;
	for(int i=0; i<nn_examples.size(); i++){
		NNConf(nn_examples[i],isin,conf,dummy);
		if(y[i]==1 && conf<= thr_nn){
			if(isin[1]<0){
				pEx = vector<Mat>(1,nn_examples[i]);
				continue;
			}
			pEx.push_back(nn_examples[i]);
		}
		if(y[i]==0&&conf>0.5)
			nEx.push_back(nn_examples[i]);
	}
	acum++;
	printf("%d. Trained NN examples: %d positive %d positive %d negative \n", acum, (int)pEx.size(),(int)nEx.size());
}

void FerNNClassifier::NNConf(const Mat& example,vector<int>& isin, float& rsconf, float& csconf){
	/*
		input:
		-NN Patch
		Putputs;
		-Relative similarity(rsconf), Conservative similarity (csconf), in pos. set|Id post set|In neg.set(isin)
	*/
	isin = vector<int>(3,1);
	if(pEx.empty()){				//if is empty(tld.pex) % IF positive examples in the model are not defiend Then everything is negative
		rsconf = 0;					//conf1 = zeros(1,size(x,2));
		csconf =0;
		return;
	}
	if(nEx.empty()){				//if isempty(tld.nex) % IF negative examples  in the mode lare not defined Then everything is positive
		rsconf = 1;					//conf1 = ones(1size(x,2));
		csconf = 1;
		return;
	}
	Mat ncc(1,1,CV_32F);
	float nccP, csmaxP, maxP = 0;
	bool anyP = false;
	int maxPidx, validatedPart = ceil(pEx.size()*valid);
	float nccN, maxN = 0;
	bool anyN = false;
	for(int i = 0;i<pEx.size(); i++){
		matchTemplate(pEx[i],example,ncc,CV_TM_CCORR_NORMED);		//measure NCC to positive examples
		nccP=(((float*)ncc.data)[0]+1)*0.5;
		if(nccP> maxP){
			maxP = nccP;
			maxPidx = i;
			if(i<validatedPart)
				csmaxP = maxP;
		}
	}
	for(int i=0; i<nEx.size(); i++){
		matchTemplate(nEx[i],example,ncc,CV_TM_CCORR_NORMED);		//measure NCC to negative examples
		nccN = (((float*)ncc.data)[0]+1)*0.5;
		if(nccN>ncc_thesame)
			anyN = true;
		if(nccN > maxN)
			maxN = nccN;
	}
	//set isin
	if(anyP) isin[0] = 1;					//if he query patch is highly correlated with any positive patch in the model then it is considered to be one of them
	isin[1] = maxPidx;						//get the index of the maximall correlated positive patch
	if(anyN) isin[2] = 1;					//if the query patch is highly correlated with any negative patch inthe model then it is condidered to be one of them
	//Measure relative similarity
	float dN = 1-maxN;
	float dP = 1-maxP;
	rsconf = (float)dN/(dN + dP);
	//Measure conservative similarity
	dP = 1-csmaxP;
	csconf = (float)dN/(dN + dP);
}

void FerNNClassifier::evaluateTh(const vector<pair<vector<int>,int>>& nXT, const vector<cv::Mat>& nExT){
	float fconf;
	for(int i=0;i<nXT.size();i++){
		fconf = (float)measure_forest(nXT[i].first)/nstructs;
		if(fconf>thr_fern)
			thr_fern = fconf;
	}
	vector <int> isin;
	float conf,dummy;
	for(int i=0; i<nExT.size();i++){
		NNConf(nExT[i],isin,conf,dummy);
		if(conf>thr_nn)
			thr_nn = conf;
	}
	if(thr_nn>  thr_nn_valid)
		thr_nn_valid = thr_nn;
}

void FerNNClassifier::show(){
	Mat examples((int)pEx.size()*pEx[0].rows,pEx[0].cols,CV_8U);
	double minval;
	Mat ex(pEx[0].rows,pEx[0].cols,pEx[0].type());
	for(int i= 0; i<pEx.size(); i++){
		minMaxLoc(pEx[i],&minval);
		pEx[i].copyTo(ex);
		ex = ex-minval;
		Mat tmp = examples.rowRange(Range(i*pEx[i].rows,(i+1)*pEx[i].rows));
		ex.convertTo(tmp,CV_8U);
	}
	imshow("Examples",examples);
}