/* Lear's GIST implementation, version 1.1, (c) INRIA 2009, Licence: PSFL */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>
#include <errno.h>
/* "readdir" etc. are defined here. */
#include <dirent.h>
/* limits.h defines "PATH_MAX". */
#include <limits.h>
#include "gist.h"

#define GIST_SIZE 960


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


static void usage(void) {
	fprintf(stderr,"compute_gist options... [infilename]\n"
			"infile is a PPM raw file\n"
			"options:\n"
			"[-nblocks nb] use a grid of nb*nb cells (default 4)\n"
			"[-orientationsPerScale o_1,..,o_n] use n scales and compute o_i orientations for scale i\n"
	);

	exit(1);
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

/* List the files in "dir_name". */
char * changeName(char *name){
	char *ppm_name = (char *)malloc(sizeof(char)*64);
	memset(ppm_name,'\0',64);
	snprintf(ppm_name,64,"%s%s",name,".ppm");
	//printf("change Name = %s \n",ppm_name);
	return ppm_name ;
}


static void
list_dir_cal_gist (const char * dir_name)
{
	DIR * d;
	struct dirent * entry;
	const char * d_name;
	int k = 0 ;
	int i = 0;
	FILE *fp=NULL ;
	char gist_file[64];
	/* Open the directory specified by "dir_name". */
	d = opendir (dir_name);

	/* Check it was opened. */
	if (! d) {
		fprintf (stderr, "Cannot open directory '%s': %s\n",
				dir_name, strerror(errno));
		exit (EXIT_FAILURE);
	}
	while((entry=readdir(d))!=NULL){
		char path[PATH_MAX];
		int path_length ;
		if (entry->d_type & DT_DIR && (strcmp (entry->d_name, "..") != 0 &&
				strcmp (entry->d_name, ".") != 0)){
			path_length = snprintf (path, PATH_MAX,
					"%s/%s", dir_name, entry->d_name);
			snprintf(gist_file,64,"%s%s.csv","./gistFeatures/",entry->d_name);
			fp=fopen(gist_file,"at");
			printf ("Dir = %s\n", path);{
				DIR * sub_d;
				struct dirent * sub_entry;
				const char * sub_d_name;
				char copy_cmd[256] ;
				char convert_cmd[256];
				char del_cmd[256];
				char *tmp_name ;
				char tmp_filepath[64];
				float gist_feature[GIST_SIZE];


				sub_d = opendir (path);
				while((sub_entry=readdir(sub_d))!=NULL){
					char sub_path[PATH_MAX];
					int sub_path_length ;
					sub_path_length = snprintf (sub_path, PATH_MAX,
							"%s/%s",path,sub_entry->d_name);
					printf("%s \n",sub_entry->d_name);
					if ((strcmp (sub_entry->d_name, "..") != 0 &&
							strcmp (sub_entry->d_name, ".") != 0)){
						printf("File = %s \n",sub_path);
						snprintf(copy_cmd,256,"%s %s %s","cp",sub_path,"./tmp");
						++k ;
						system(copy_cmd);
						tmp_name = changeName(sub_entry->d_name);
						printf("Tmp Name = %s \n",tmp_name);
						snprintf(tmp_filepath,64,"%s%s","./tmp/",tmp_name);
						snprintf(convert_cmd,256,"%s%s %s","convert ./tmp/",sub_entry->d_name,tmp_filepath);
						printf("convert_cmd = %s \n",convert_cmd);
						system(convert_cmd);
						cal_GistFeature(tmp_filepath,gist_feature);


						for(i=0;i<GIST_SIZE;i++){
							fprintf(fp,"%.4f,",gist_feature[i]);
						}
						fputc('\r',fp);
						fputc('\n',fp);
						snprintf(del_cmd,256,"%s %s/%s","rm","./tmp",sub_entry->d_name);
						printf("%s \n",del_cmd);
						system(del_cmd);
						free(tmp_name);

					}
				}
			}
			//	printf ("%s\n", entry->d_name);
			fclose(fp);
		}

	}

	printf("K= %d",k);

}

int main(){
//	cal_GistFeature("./ar.ppm");
	list_dir_cal_gist ("./dataset");
	//list_dir_cal_gist ("/home/kumar/computer_vision_class/101_ObjectCategories");

	return 0 ;
}
