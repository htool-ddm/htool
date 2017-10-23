#include "htool/cluster.hpp"

using namespace std;
using namespace htool;

int main(int argc, char const *argv[]) {

  SetMinClusterSize(1);
  bool test =0;

  int size = 10;
	double z = 1;
	vector<R3>     p(size);
  vector<double> r(size,0);
  vector<double> g(size,1);
	vector<int>    tab(size);
  vector<int>    perm(size);
	for(int j=0; j<size; j++){
		double rho = ((double) rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
		double theta = ((double) rand() / (double)(RAND_MAX));
		p[j][0] = sqrt(rho)*cos(2*M_PI*theta); p[j][1] = sqrt(rho)*sin(2*M_PI*theta); p[j][2] = z;
		// sqrt(rho) otherwise the points would be concentrated in the center of the disk
		tab[j]=j;
	}


  Cluster t(p,r,tab,g,perm);

  // Test
  std::stack<Cluster*> s;
  s.push(&t);
  int depth =0;
  while (!s.empty()){
    Cluster* curr = s.top();
    s.pop();

    if (!curr->IsLeaf()){
      // test num inclusion

      // Offset of son 0 = offset of curr
      test = test || !(curr->get_offset()==curr->get_son(0).get_offset());

      // Offset of curr + size of son 0 = offset of son 1
      test = test || !(curr->get_offset()+curr->get_son(0).get_size()==curr->get_son(1).get_offset());
      // Offset of curr + its size = offset of son 1 + its size
      test = test || !(curr->get_offset()+curr->get_size()==curr->get_son(1).get_offset()+curr->get_son(1).get_size());
      s.push(&(curr->get_son(0)));
			s.push(&(curr->get_son(1)));
    }

  }
  cout<<"max depth : "<<t.get_max_depth()<<endl;
  cout<<"min depth : "<<t.get_min_depth()<<endl;

  test = test || !(t.get_max_depth()==4 && t.get_min_depth()==3);
  t.print(perm);


  std::cout << "test "<< test << std::endl;
  return test;
}
