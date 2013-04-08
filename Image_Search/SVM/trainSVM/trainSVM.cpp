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
extern "C" {
#include "gist.h"
}

using namespace std ;
using namespace cv ;
#include "csv_parser.hpp"

#define GIST_FT_FOLDER "./../imageSearch/gistFeatures/"
#define MAX_FILES 10
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define NO_ATTRIB 960

char gist_fileNames[MAX_FILES][64];
char map_filenames[MAX_FILES][32] ;

vector< vector <float > > gistft_vector ;
vector <float > gist_class ;


void setgist_fileNames(){
    DIR *d ;
    struct dirent *entry;
    char * f_name ;
    int i = 0 ;
    memset(gist_fileNames,'\0',(MAX_FILES * 64));
    d = opendir(GIST_FT_FOLDER);
    while((entry=readdir(d))){
        f_name = entry->d_name;
        if(strcmp(f_name,".") && strcmp(f_name,"..")){
            snprintf(gist_fileNames[i],64,"%s%s",GIST_FT_FOLDER,f_name);
            snprintf(map_filenames[i],32,"%s",f_name);
            //cout<<gist_fileNames[i]<<endl;
            ++i ;
        }

    }
}

int numberof_TraningData(){
    int i = 0 ;
    int result= 0;
    while(gist_fileNames[i][0]!='\0'){
        csv_parser csv(gist_fileNames[i]);
        result+=(csv.total_lines()-1);
        ++i ;
    }
    return result ;
}

void calVectors(){
    int i =0 ;
    vector <float > tmp ;
    while(gist_fileNames[i][0]!='\0'){
        cout<<gist_fileNames[i]<<endl;

        csv_parser csv(gist_fileNames[i]);
        int totallines=csv.total_lines()-1;
        cout<<"Lines"<<totallines<<endl;
        for(int j=1 ;j<=totallines;j++){
            for(int k=1 ; k<=NO_ATTRIB ;k++ ){
                tmp.push_back(atof(csv.get_value(j,k).c_str()));
            }
            gistft_vector.push_back(tmp);
            gist_class.push_back(i);
            tmp.clear();
        }


        i++ ;
    }
    //cout<<gistft_vector.size()<<gistft_vector[0].size()<<endl;
    //cout<<gistft_vector.at(0).at(0)<<endl;
    for(int l = 0 ; l < gist_class.size() ;l++){
        gist_class.at(l);
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
    FILE *fd;
    filePath=argv[1];
    Mat classes;
    Mat trainingData;

    setgist_fileNames();
    calVectors();

    trainingData = cvCreateMat(gistft_vector.size(), gistft_vector[0].size(),CV_32FC1);
    for(int i=0; i<gistft_vector.size(); i++) {
        for(int j=0 ; j<gistft_vector[i].size(); j++) {
            trainingData.at<float>(i,j)=gistft_vector[i][j];
        }
    }
    Mat(gist_class).copyTo(classes);

    // cout<<trainingData<<endl;

    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
    cout<<"Train the SVM"<<endl;
    CvSVM SVM;

    fd=fopen("SVMClassifier.xml","r");
    if(fd!=NULL){
        cout<<"Model file already exit"<<endl;
        fclose(fd);
        return 0;
    }
    else{
        cout<<"Trainning the SVM"<<endl;
        SVM.train(trainingData, classes, Mat(), Mat(), params);
        SVM.save("SVMClassifier.xml");
        FILE *mfp ;
        mfp=fopen("map_file.csv","wt");
        int i=0 ;
        char tmp[32];
        while(map_filenames[i][0]!='\0'){
            memset (tmp,'\0',32);
            strncpy(tmp,map_filenames[i],strlen(map_filenames[i])-4);
            fprintf(mfp,"%d,%s,",i,tmp);
            fputc('\r',mfp);
            fputc('\n',mfp);
            i++ ;
        }
        fclose(mfp);

    }

#if 0	
    cout<<"SVM Predict"<<endl;
    float response = SVM.predict(sampleMat);
    cout<<"SVM RESPONCE"<<response<<endl;
#endif
    return 0;
}
