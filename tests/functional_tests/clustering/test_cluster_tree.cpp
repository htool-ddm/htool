#include <htool/clustering/cluster_tree.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
  MPI_Init(&argc,&argv);
  SetMinClusterSize(1);
  bool test =0;

  int size = 10;
	double z = 1;
	vector<R3>     p(size);
  vector<double> r(size,0);
  vector<double> g(size,1);
	vector<int>    tab(size);

	for(int j=0; j<size; j++){
		double rho = ((double) rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
		double theta = ((double) rand() / (double)(RAND_MAX));
		p[j][0] = sqrt(rho)*cos(2*M_PI*theta); p[j][1] = sqrt(rho)*sin(2*M_PI*theta); p[j][2] = z;
		// sqrt(rho) otherwise the points would be concentrated in the center of the disk
		tab[j]=j;
	}

  Cluster_tree t(p,r,tab,g);
  t.print();

  std::vector<int> vector_test_1(size),vector_test_2(size),vector_test_3(size);
  std::iota(vector_test_1.begin(),vector_test_1.end(),0);
  t.global_to_cluster(vector_test_1.data(),vector_test_2.data());
  t.cluster_to_global(vector_test_2.data(),vector_test_3.data());
  std::cout << "test 1 :"<<std::endl;
  std::cout << vector_test_1 << std::endl;
  std::cout << "test 2 :"<<std::endl;
  std::cout << vector_test_2 << std::endl;
  std::cout << "test 3 :"<<std::endl;
  std::cout << vector_test_3 << std::endl;
  test = test || !(vector_test_1==vector_test_3);

  std::cout << t.get_labels(2) << std::endl;




  MPI_Finalize();
  std::cout << "test : "<<test << std::endl;
  return test;
}
