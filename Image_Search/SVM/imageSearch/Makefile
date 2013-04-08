# Makefile to build imageSearchMethod Program
# --- macros
CC=gcc
#CFLAGS=  -O3 -I /usr/local/lib/sprng/include -I /usr/local/lib/pgplot -g
OBJECTS= imageSearchMethod.o standaloneImage.o gist.o 
#LIBS = -L/usr/local/lib/sprng/lib -llcg -L/usr/local/lib/pgplot -lcpgplot -lpgplot -lX11 -lftn -lm
LIBS = -lfftw3f


# --- targets
all:    imageSearch
imageSearch: $(OBJECTS) 
	$(CC) -g -o imageSearchMethod  $(OBJECTS) $(LIBS)
        
standaloneImage.o: standaloneImage.c
	$(CC) $(CFLAGS) -c standaloneImage.c
       
gist.o: gist.c
	$(CC) $(CFLAGS) -c gist.c -DUSE_GIST -DSTANDALONE_GIST

imageSearchMethod.o: imageSearchMethod.c
	$(CC) $(CFLAGS) -c imageSearchMethod.c -DUSE_GIST -DSTANDALONE_GIST

# --- remove binary and executable files
clean:
	rm -f imageSearch *.o $(OBJECTS)
