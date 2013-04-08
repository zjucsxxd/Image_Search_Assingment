//============================================================================
// Name        : Bow.cpp
// Author      : Kumar Vishal
// Version     :
// Copyright   : rising_sun
// Description : Hello World in C, Ansi-style
//============================================================================

#include<stdio.h>
/* "readdir" etc. are defined here. */
#include <dirent.h>
/* limits.h defines "PATH_MAX". */
#include <limits.h>
#include <vector>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <map>
using namespace std;
using namespace boost::filesystem;
using namespace cv;

//location of the training data
#define TRAINING_DATA_DIR "data/train"
//location of the evaluation data
#define EVAL_DATA_DIR "data/eval"

Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");

int dictionarySize = 1000;
TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;

map<int, string> label_name ;

BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
BOWImgDescriptorExtractor bowDE(extractor, matcher);

#define MAX_DIR 10
char dir_list[MAX_DIR][32] ;

void extractTrainingVocabulary(const path& basepath) {
	for (directory_iterator iter = directory_iterator(basepath); iter
			!= directory_iterator(); iter++) {
		directory_entry entry = *iter;

		if (is_directory(entry.path())) {

			cout << "Processing directory " << entry.path().string() << endl;
			extractTrainingVocabulary(entry.path());

		} else {

			path entryPath = entry.path();
			if (entryPath.extension() == ".jpg") {

				//cout << "Processing file " << entryPath.string() << endl;
				Mat img = imread(entryPath.string());
				if (!img.empty()) {
					vector<KeyPoint> keypoints;
					detector->detect(img, keypoints);
					if (keypoints.empty()) {
						cerr << "Warning: Could not find key points in image: "
								<< entryPath.string() << endl;
					} else {
						Mat features;
						extractor->compute(img, keypoints, features);
						bowTrainer.add(features);
					}
				} else {
					cerr << "Warning: Could not read image: "
							<< entryPath.string() << endl;
				}

			}
		}
	}
}

 void fill_DirNAmes(char *dir_name){
	DIR * d;
 	FILE *fp=NULL ;
	struct dirent * entry;
	//const char * d_name;
	
	memset(dir_list[0],'\0',MAX_DIR*32);
	/* Open the directory specified by "dir_name". */
	d = opendir(dir_name);

	/* Check it was opened. */
	if (! d) {
		fprintf (stderr, "Cannot open directory '%s': %s\n",
				dir_name, strerror(errno));
		exit (EXIT_FAILURE);
	}
	int i = 0 ;
	while((entry=readdir(d))!=NULL){
	if (entry->d_type & DT_DIR && (strcmp (entry->d_name, "..") != 0 &&
				strcmp (entry->d_name, ".") != 0)){
				strncpy(dir_list[i],entry->d_name,32);
				label_name[i]=entry->d_name ;
				i++ ;
		}
	}
	i=0;
	while(dir_list[i][0]!='\0'){
		cout<<dir_list[i]<<endl;
		i++ ;
	}
	closedir(d);
 }

 void my_extractBOWDescriptor(char *dir_name , Mat& trainingData, Mat& labels){
 	fill_DirNAmes(dir_name);
 	cout<<"my_extractBOWDescriptor"<<endl;
 	int i= 0 ;
 	DIR * d;
 	FILE *fp=NULL ;
	struct dirent * entry;
	//const char * d_name;
 	char tmp_dir[64];
 	char tmp_filename[128];
 	while(dir_list[i][0]!='\0'){
 		memset(tmp_dir,'\0',64);
 		snprintf(tmp_dir,64,"%s/%s/",TRAINING_DATA_DIR,dir_list[i]);
 		cout<<tmp_dir<<endl;
 		d = opendir(tmp_dir);
 		while((entry=readdir(d))!=NULL){
			if ((strcmp (entry->d_name, "..") != 0 && strcmp (entry->d_name, ".") != 0)){
					cout<<"file"<<entry->d_name<<endl;
					memset(tmp_filename,'\0',128);
					snprintf(tmp_filename,128,"%s%s",tmp_dir,entry->d_name);
					cout<<"tmp"<<tmp_filename<<endl;
					/***/
					
					Mat img = imread(tmp_filename);
					if (!img.empty()) {
						vector<KeyPoint> keypoints;
						detector->detect(img, keypoints);
						if (keypoints.empty()) {
							cerr << "Warning: Could not find key points in image: " <<endl;
					} else {
						Mat bowDescriptor;
						bowDE.compute(img, keypoints, bowDescriptor);
						trainingData.push_back(bowDescriptor);
						float label=(float)(i);
						cout<<"Label ="<<i<<endl;
						labels.push_back(label);
					}
				} else {
					cerr << "Warning: Could not read image: " << endl;
					}
			
					
					/***/
					
				}
 		
 		}
 		closedir(d);
 	i++ ;
  }
  cout<<"my_extractBOWDescriptor_exit"<<endl;
} 
#if 1
void my_extractTestBOWDescriptor(char *file_name,Mat& testData){
	
	Mat img = imread(file_name);
	if (!img.empty()) {
		vector<KeyPoint> keypoints;
		detector->detect(img, keypoints);
			if (keypoints.empty()) {
				cerr << "Warning: Could not find key points in image: " << file_name << endl;
			}else{
				Mat bowDescriptor;
				bowDE.compute(img, keypoints, bowDescriptor);
				testData.push_back(bowDescriptor);
				}
			}else{
					cerr << "Warning: Could not read image: " << file_name << endl;
			}
		
}
#endif

int main(int argc, char ** argv) {

	
	
	FILE *fd ;
	char fileName[128] ;
	memset(fileName,'\0',128);
	if ( argc != 2 ) {
        	/* We print argv[0] assuming it is the program name */
        	printf( "usage: %s filename", argv[0] );
        	return -1 ;
    	}
    	else {
		fd=fopen(argv[1],"r");
    		if(fd==NULL){
    			cout<<"File path In Correct"<<endl;
    			return -1 ;
    		}
    		else{
    			strncpy(fileName,argv[1],128);
    			fclose(fd);
    		}
    	}		
    
	cout<<"Creating dictionary..."<<endl;
    
	extractTrainingVocabulary(path(TRAINING_DATA_DIR));
	vector<Mat> descriptors = bowTrainer.getDescriptors();
	
	
	
	
	int count=0;
	for(vector<Mat>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
	{
		count+=iter->rows;
	}
	cout<<"Clustering "<<count<<" features"<<endl;
	Mat dictionary = bowTrainer.cluster();
	bowDE.setVocabulary(dictionary);
	cout<<"Processing training data..."<<endl;
	Mat trainingData(0, dictionarySize, CV_32FC1);
	Mat labels(0, 1, CV_32FC1);
	my_extractBOWDescriptor(TRAINING_DATA_DIR,trainingData,labels);
	//extractBOWDescriptor(path(TRAINING_DATA_DIR), trainingData, labels);
	
	NormalBayesClassifier classifier;
	cout<<"Training classifier..."<<endl;

	fd=fopen("NBClassifier.xml","r");
	if(fd!=NULL){
		classifier.load("NBClassifier.xml");
		fclose(fd);
	}
	else{
		classifier.train(trainingData, labels);
		classifier.save("NBClassifier.xml");
	}
	
	Mat fillTestData(0, dictionarySize, CV_32FC1);
	my_extractTestBOWDescriptor(fileName,fillTestData);
	int result = (int)classifier.predict(fillTestData) ;
	cout<<"Result = "<<result<<endl;
	cout<<"Label = "<<label_name.find(result)->second<<endl;
	
}
