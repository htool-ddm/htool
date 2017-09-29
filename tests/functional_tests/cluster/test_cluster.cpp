#include "htool/cluster.hpp"

using namespace std;
using namespace htool;

int main(int argc, char const *argv[]) {

  SetMinClusterSize(1);
  bool test =0;

	int n1 = 5;
  int n2 = 5;
  int nr = n1+n2;
	double z = 1;
	vector<R3>     p(nr);
	vector<int>    tab(nr);
	for(int j=0; j<n1; j++){
		double rho = ((double) rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
		double theta = ((double) rand() / (double)(RAND_MAX));
		p[j][0] = sqrt(rho)*cos(2*M_PI*theta); p[j][1] = sqrt(rho)*sin(2*M_PI*theta); p[j][2] = z;
		// sqrt(rho) otherwise the points would be concentrated in the center of the disk
		tab[j]=j;
	}

  double dist = 5;
  for(int j=n1; j<nr; j++){
		double rho = ((double) rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
		double theta = ((double) rand() / (double)(RAND_MAX));
		p[j][0] = sqrt(rho)*cos(2*M_PI*theta); p[j][1] = sqrt(rho)*sin(2*M_PI*theta); p[j][2] = z + dist;
		// sqrt(rho) otherwise the points would be concentrated in the center of the disk
		tab[j]=j;
	}

  Cluster t(p,tab);
  t.build();

  // Test
  Cluster* s = &t;
  int depth =0;
  while (s!=NULL){
    // test depth
    test = test || !((*s).get_depth()==depth);

    depth+=1;
    if ((*s).IsLeaf()){
      s=NULL;
    }
    else{
      // test num inclusion
      if (!((*s).get_son(0).IsLeaf()) && !((*s).get_son(1).IsLeaf())){
        std::vector<int> root = (*s).get_num();
        std::vector<int> son0 = (*s).get_son(0).get_num();
        std::vector<int> son1 = (*s).get_son(1).get_num();
        for (int i=0;i<root.size();i++){
          int count0 = count(son0.begin(),son0.end(),root[i]);
          int count1 = count(son1.begin(),son1.end(),root[i]);
          test = test || !((count0==0 && count1==1) || (count0==1 && count1==0) );
        }
      }
      s=&((*s).get_son(0));
    }
  }
  cout<<"max depth : "<<t.get_max_depth()<<endl;
  cout<<"min depth : "<<t.get_min_depth()<<endl;

  test = test || !(t.get_max_depth()==4 && t.get_min_depth()==3);
  t.print_offset();



  return test;
}
