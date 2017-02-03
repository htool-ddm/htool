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

  Matrix   A;
  LoadMatrix(matrixname.c_str(),A);
  matrix_to_bytes(A, (matrixname.substr(0,matrixname.size()-4)+".bin").c_str());

  return 0;
}
