#include "htool/point.hpp"
#include <iostream>
using namespace std;
using namespace htool;
int main(int argc, char const *argv[]) {
  N2 a = {1,2};
  N2 b = {2,1};
  cout << a+b << endl;
  cout << a - b << endl;
  cout << 3*a << endl;
  a*=0;
  cout << a << endl;
  a+=b;
  cout<< a << endl;
  a*=3;
  cout<< a << endl;
  R3 c{};
  cout << c << endl;
  long double d =10;
  cout<< typeid(d*0).name()<<endl;
}
