#include <htool/fullACA.hpp>
#include <htool/hmatrix.hpp>
#include <htool/loading.hpp>

using namespace std;
using namespace htool;


class MyMatrix: public IMatrix<double>{
	const vector<R3>& p1;
	const vector<R3>& p2;

public:
	MyMatrix(const vector<R3>& p10,const vector<R3>& p20 ):IMatrix(p10.size(),p20.size()),p1(p10),p2(p20) {}

	double get_coef(const int& i, const int& j)const {return 1./(4*M_PI*norm(p1[i]-p2[j]));}


  std::vector<double> operator*(std::vector<double> a){
		std::vector<double> result(p1.size(),0);
		for (int i=0;i<p1.size();i++){
			for (int k=0;k<p2.size();k++){
				result[i]+=this->get_coef(i,k)*a[k];
			}
		}
		return result;
	 }
};


int main(int argc, char const *argv[]) {
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //
  bool test = 0;
  const int ndistance = 4;
  double distance[ndistance];
  distance[0] = 3; distance[1] = 5; distance[2] = 7; distance[3] = 10;
  SetNdofPerElt(1);
  SetEpsilon(1e-6);
  SetEta(0.1);
  SetMinClusterSize(100);
  typedef double scalar_type;



  	for(int idist=0; idist<ndistance; idist++)
  	{
  		cout << "Distance between the clusters: " << distance[idist] << endl;

  		srand (1);
  		// we set a constant seed for rand because we want always the same result if we run the check many times
  		// (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

  		int nr = 2000;
  		int nc = 1000;
  		vector<int> Ir(nr); // row indices for the lrmatrix
  		vector<int> Ic(nc); // column indices for the lrmatrix

  		double z1 = 1;
  		vector<R3>     p1(nr);
  		vector<int>  tab1(nr);
  		for(int j=0; j<nr; j++){
  			Ir[j] = j;
  			double rho = ((double) rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
  			double theta = ((double) rand() / (double)(RAND_MAX));
  			p1[j][0] = sqrt(rho)*cos(2*M_PI*theta); p1[j][1] = sqrt(rho)*sin(2*M_PI*theta); p1[j][2] = z1;
  			// sqrt(rho) otherwise the points would be concentrated in the center of the disk
  			tab1[j]=j;
  		}
  		// p2: points in a unit disk of the plane z=z2
  		double z2 = 1+distance[idist];
  		vector<R3> p2(nc);
  		vector<int> tab2(nc);
  		for(int j=0; j<nc; j++){
              Ic[j] = j;
  			double rho = ((double) rand() / (RAND_MAX)); // (double) otherwise integer division!
  			double theta = ((double) rand() / (RAND_MAX));
  			p2[j][0] = sqrt(rho)*cos(2*M_PI*theta); p2[j][1] = sqrt(rho)*sin(2*M_PI*theta); p2[j][2] = z2;
  			tab2[j]=j;
  		}

  		MyMatrix A(p1,p2);


  		HMatrix<fullACA,scalar_type> HA(A,p1,tab1,p2,tab2);
  		int nbr_dmat = HA.get_ndmat();
  		int nbr_lmat = HA.get_nlrmat();
  		double comp = HA.compression();

  		if (rank==0){
  			cout << "nbr_dmat : "<<nbr_dmat<<endl;
  			cout << "nbr_lmat : "<<nbr_lmat<<endl;
  			cout << "compression : "<<comp<<endl;
  		}
  		HA.print_stats();
    	double mytime, maxtime, meantime;


      // Global vectors
  		std::vector<scalar_type> x_global(nc,1),f_global(nr),f_global_test(nr);

      // Local vectors
  		int nr_local=HA.get_local_size_cluster();
  		const std::vector<std::vector<int>>& MasterClusters = HA.get_MasterClusters();
  		std::vector<scalar_type> x_local(nr_local,1),f_local(nr_local),f_local_test(nr_local);


      // Global product
      HA.mvprod_global(x_global.data(),f_global.data());

  		// Local product
      HA.mvprod_local(x_local.data(),f_local.data());



      // Global to local
  		for (int i=0; i< MasterClusters[rank].size(); i++) {
  			f_local_test[i] = f_global[MasterClusters[rank][i]];
  		}

      // Local to global
      std::vector<double> rcv;
      rcv.resize(nr);

      std::vector<int> recvcounts(size);
      std::vector<int>  displs(size);

      displs[0] = 0;

      for (int i=0; i<size; i++) {
      	recvcounts[i] = MasterClusters[i].size();
      	if (i > 0)
      		displs[i] = displs[i-1] + recvcounts[i-1];
      }


      MPI_Allgatherv(&(f_local.front()), recvcounts[rank], MPI_DOUBLE, &(rcv.front()), &(recvcounts.front()), &(displs.front()), MPI_DOUBLE, HA.get_comm());

      for (int i=0; i<size; i++)
      for (int j=0; j< MasterClusters[i].size(); j++)
      	f_global_test[MasterClusters[i][j]] = rcv[displs[i]+j];

      // Errors
      double global_diff = norm2(f_global-f_global_test)/norm2(f_global);
      double local_diff = norm2(f_local-f_local_test);
      MPI_Allreduce(MPI_IN_PLACE, &local_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      local_diff/=norm2(f_global);

  		if (rank==0){
  			cout <<"difference on mat vec prod computed globally: "<<global_diff << endl;
        cout <<"difference on mat vec prod computed locally: "<<local_diff << endl;
  		}

  		test = test || !(local_diff<1.e-15 && global_diff<1.e-15); // default tol in hpddm

  	}
  	// Finalize the MPI environment.
  	MPI_Finalize();

  	return test;
  }
