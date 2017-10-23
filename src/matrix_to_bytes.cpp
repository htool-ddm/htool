#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <cassert>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <mpi.h>
#include <htool/htool.hpp>
#include <htool/loading.hpp>
using namespace std;
using namespace htool;

int main(int argc, char const *argv[]) {
  // Check the number of parameters
  if (argc < 2) {
    // Tell the user how to run the program
    cerr << "Usage: " << argv[0] << " matrix name" << endl;
    /* "Usage messages" are a conventional way of telling the user
     * how to run a program if they enter the command incorrectly.
     */
    return 1;
  }
  string matrixname = argv[1];
  SetNdofPerElt(3);
  Matrix<double>   A;
  LoadMatrix(matrixname.c_str(),A);
  for (int i=0;i<10;i++){
    for(int j=0;j<10;j++){
      cout << i <<" "<<j<< " "<<A(i,j) << endl;
    }
  }
  A.matrix_to_bytes( (matrixname.substr(0,matrixname.size()-4)+".bin").c_str());

  return 0;
}
