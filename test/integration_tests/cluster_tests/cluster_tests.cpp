#include "htool/cluster.hpp"

using namespace std;
using namespace htool;

int main(int argc, char const *argv[]) {

  SetMinClusterSize(1);


	int n1 = 5;
  int n2 = 5;
  int nr = n1+n2;
	double z = 1;
	vector<R3>     p(nr);
	vector<double> r(nr);
	vector<int>    tab(nr);
	for(int j=0; j<n1; j++){
		double rho = ((double) rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
		double theta = ((double) rand() / (double)(RAND_MAX));
		p[j][0] = sqrt(rho)*cos(2*M_PI*theta); p[j][1] = sqrt(rho)*sin(2*M_PI*theta); p[j][2] = z;
		// sqrt(rho) otherwise the points would be concentrated in the center of the disk
		r[j]=0.;
		tab[j]=j;
	}

  double dist = 5;
  for(int j=n1; j<nr; j++){
		double rho = ((double) rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
		double theta = ((double) rand() / (double)(RAND_MAX));
		p[j][0] = sqrt(rho)*cos(2*M_PI*theta); p[j][1] = sqrt(rho)*sin(2*M_PI*theta); p[j][2] = z + dist;
		// sqrt(rho) otherwise the points would be concentrated in the center of the disk
		r[j]=0.;
		tab[j]=j;
	}

  Cluster t(p,r,tab);
  t.build();

  return 0;
}
