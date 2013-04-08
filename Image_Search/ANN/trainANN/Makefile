#Makefile to compile using Aria

CC 	= g++
FLAGS 	= -Wall -g
#LIB 	= -L /usr/local/Aria/lib/ -lAria -ldl -lpthread -lrt 
#INCLUDE	= -I /usr/local/Aria/include/
LIB = -lgist -lfftw3f
INCLUDE =
all: train 

train: csv_parser.o trainANN.o 
	$(CC) -o $@  csv_parser.o trainANN.o $(LIB) `pkg-config --cflags --libs opencv`
# "$@" is the variable for the ouput
# "$?" is the variable for the input

csv_parser.o: csv_parser.cpp 
	$(CC) -c csv_parser.cpp $(INCLUDE)

#If pkg-config is giving error, follow instruction in Readme.txt

#Uncomment the below two lines if opencv is installed in the system

trainANN.o: trainANN.cpp 
	$(CC) -c $? `pkg-config --cflags --libs opencv`

clean:
	rm -rf *.o
	rm -rf train
