//============================================================================
// Name        : csv_parser.hpp
// Author      : Kumar Vishal
// Version     :
// Copyright   : rising_sun
// Description : Hello World in C, Ansi-style
//============================================================================
#ifndef __CSV_PARSER_H__
#define __CSV_PARSER_H__

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

class csv_parser
{
private:
  std::ifstream csv_file;


public:
  csv_parser(std::string filename);
  ~csv_parser();
  std::string get_line(int line_number);//Returns entire line as a string based on line number.
  int fields(std::string line); //Returns Number of fields in the line
  std::string get_value(int row,int column); //Returns the field in the specified row and column.
  int total_lines();
};

#endif
