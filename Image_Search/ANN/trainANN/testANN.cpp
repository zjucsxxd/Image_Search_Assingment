//============================================================================
// Name        : TrainImage_SVM.cpp
// Author      : Kumar Vishal
// Version     :
// Copyright   : rising_sun
// Description : Hello World in C, Ansi-style
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <dirent.h>
#include <iostream>
#include <vector>
#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <vector>
#include <map>
#include "csv_parser.hpp"
extern "C" {
#include "gist.h"
}

using namespace std ;
using namespace cv ;

#define NO_ATTRIB 960
#define MAX_FILES 10
#define NO_OF_CLASS 5
char gist_fileNames[MAX_FILES][64];

vector< vector <float > > gistft_vector ;
vector <float > gist_class ;

map<int, string> class_fileMap;

void fill_FileClassMap(){
	csv_parser csv("map_file.csv");
	int noRows=csv.total_lines()-1 ;
	int className = -1;
	string fileName ;
	for(int i=1;i<=noRows;i++){
		className=atoi(csv.get_value(i,1).c_str());
		fileName=csv.get_value(i,2);	
		class_fileMap[className]=fileName;
	}

}
static color_image_t *load_ppm(const char *fname) {
	FILE *f=fopen(fname,"r");
	if(!f) {
		perror("could not open infile");
		exit(1);
	}
	int px,width,height,maxval;
	if(fscanf(f,"P%d %d %d %d",&px,&width,&height,&maxval)!=4 ||
			maxval!=255 || (px!=6 && px!=5)) {
		fprintf(stderr,"Error: input not a raw PGM/PPM with maxval 255\n");
		exit(1);
	}
	fgetc(f); /* eat the newline */
	color_image_t *im=color_image_new(width,height);

	int i;
	for(i=0;i<width*height;i++) {
		im->c1[i]=fgetc(f);
		if(px==6) {
			im->c2[i]=fgetc(f);
			im->c3[i]=fgetc(f);
		} else {
			im->c2[i]=im->c1[i];
			im->c3[i]=im->c1[i];
		}
	}

	fclose(f);
	return im;
}

int cal_GistFeature(char *filepath,float *gist_feature) {

	const char *infilename= filepath;
	int nblocks=4;
	int n_scale=3;
	int orientations_per_scale[50]={8,8,4};
	color_image_t *im=load_ppm(infilename);
	float *desc=color_gist_scaletab(im,nblocks,n_scale,orientations_per_scale);
	int i;
	int descsize=0;
	/* compute descriptor size */
	for(i=0;i<n_scale;i++)
		descsize+=nblocks*nblocks*orientations_per_scale[i];

	descsize*=3; /* color */

	/* print descriptor */
	for(i=0;i<descsize;i++){
		gist_feature[i]=desc[i];
		//printf("%.4f ",desc[i]);
	}
	printf("\n");
	printf("Descriptor Size= %d\n",descsize);
	free(desc);

	color_image_delete(im);

	return 0;
}
char * changeName(char *name){
	char *ppm_name = (char *)malloc(sizeof(char)*64);
	memset(ppm_name,'\0',64);
	snprintf(ppm_name,64,"%s%s",name,".ppm");
	//printf("change Name = %s \n",ppm_name);
	return ppm_name ;
}

char * createppm_file(char *file){
	char convert_cmd[256] ;
	char * tmp_file=changeName(file);
	snprintf(convert_cmd,256,"%s%s %s","convert ",file,tmp_file);
	system(convert_cmd);
	return tmp_file ;
}

int main ( int argc, char *argv[] )
{

	char *filePath ;
	FILE *fd ;
	filePath=argv[1];
	Mat classes;
	Mat trainingData;


	float gistDescriptor[960];
	if ( argc != 2 ) /* argc should be 2 for correct execution */
	{
		/* We print argv[0] assuming it is the program name */
		printf( "usage: %s filename", argv[0] );
		return -1 ;
	}
	else{
		FILE *fp =NULL ;
				filePath=argv[1];
		fp=fopen(filePath,"r");
		if(fp==NULL){
			cout<<"File Doesn't Exist Exit Program"<<endl;
			exit(-1);
		}
		fclose(fp);
	}
	char *ppm_filepath=createppm_file(filePath);
	cal_GistFeature(ppm_filepath,gistDescriptor);
	free(ppm_filepath);
        //Mat sampleMat(960, 1, CV_32FC1, gistDescriptor);
        Mat sampleMat(1, 960, CV_32FC1, gistDescriptor);
	/* Fill map Data */
	
	fill_FileClassMap();
	// Train the SVM
	cout<<"Test the ANN"<<endl;

        // Set up ANN's parameters

    Mat layer_sizes( 1, 3, CV_32SC1 );
    layer_sizes.at<int>(0) = 960;
    layer_sizes.at<int>(1) = 5;
    layer_sizes.at<int>(2) = NO_OF_CLASS; /* Number of tarining folder*/

   // ANN constructor
    CvANN_MLP  ann( layer_sizes, CvANN_MLP::SIGMOID_SYM, 1, 1 );

	fd=fopen("ANNClassifier.xml","r");
	if(fd!=NULL){
		cout<<"Load Classifier"<<endl;
		ann.load("ANNClassifier.xml");
		fclose(fd);
	}
	else{
		cout<<"Model file not found"<<endl;		
                return 0;
	}
	

        Mat outputs( 1, 2, CV_32FC1, sampleMat.data );
    cout<<outputs<<endl;
    ann.predict(sampleMat, outputs);
    Point maxLoc;
    minMaxLoc(outputs, 0, 0, 0, &maxLoc );
    cout<<"ANN RESPONCE"<<maxLoc.x<<endl;

	cout<<"ANN Response Index := "<<maxLoc.x<<endl;
	cout<<"**** "<<class_fileMap.find((int)maxLoc.x)->second<<"  ****"<<endl;
	return 0;
}
