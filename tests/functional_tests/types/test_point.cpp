#include "htool/types/point.hpp"
#include <iostream>
using namespace std;
using namespace htool;
int main(int argc, char const *argv[]) {
  bool test = 0;

  // int
  N2 ai = {1,2};
  N2 bi = {2,1};
  N2 ci = {3,3};
  N2 di = {-1,1};
  N2 ei = {3,6};
  N2 fi = {0,0};
  N2 gi = {6,3};

  test = test || !(ai+bi==ci);
  cout << ai+bi << endl;
  test = test || !(ai-bi==di);
  cout << ai - bi << endl;
  test = test || !(3*ai==ei);
  cout << 3*ai << endl;
  ai*=0;
  test = test || !(ai==fi);
  cout << ai << endl;
  ai+=bi;
  test = test || !(ai==bi);
  cout<< ai << endl;
  ai*=3;
  test = test || !(ai==gi);
  cout<< ai << endl;

  // double
  R2 ad = {1,2};
  R2 bd = {2,1};
  R2 cd = {3,3};
  R2 dd = {-1,1};
  R2 ed = {3,6};
  R2 fd = {0,0};
  R2 gd = {6,3};

  test = test || !(norm2(ad+bd-cd)<1e-16);
  cout << ad+bd << endl;
  test = test || !(norm2(ad-bd-dd)<1e-16);
  cout << ad - bd << endl;
  test = test || !(norm2(3.*ad-ed)<1e-16);
  cout << 3.*ad << endl;
  ad*=0.;
  test = test || !(norm2(ad-fd)<1e-16);
  cout << ad << endl;
  ad+=bd;
  test = test || !(norm2(ad-bd)<1e-16);
  cout<< ad << endl;
  ad*=3.;
  test = test || !(ad==gd);
  cout<< ad << endl;


  return test;
}
