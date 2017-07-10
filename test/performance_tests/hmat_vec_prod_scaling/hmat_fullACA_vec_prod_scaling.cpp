#include <htool/fullACA.hpp>
#include <htool/matrix.hpp>
#include <htool/hmatrix.hpp>

using namespace std;
using namespace htool;


class MyMatrix: public IMatrix<double>{
	const vector<R3>& p1;
	const vector<R3>& p2;

public:
	MyMatrix(const vector<R3>& p10,const vector<R3>& p20 ):IMatrix(p10.size(),p20.size()),p1(p10),p2(p20) {}

	double get_coef(const int& i, const int& j)const {return 1./(4*M_PI*norm2(p1[i]-p2[j]));}


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


int main(int argc, char const *argv[]){

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
	double distance=3;

	SetNdofPerElt(3);
	SetEpsilon(1e-6);
	SetEta(0.1);
	SetMinClusterSize(100);

  // Create points randomly
	srand (1);
	// we set a constant seed for rand because we want always the same result if we run the check many times
	// (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

	int nr = 20000;
	int nc = 10000;
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
	double z2 = 1+distance;
	vector<R3> p2(nc);
	vector<int> tab2(nc);
	for(int j=0; j<nc; j++){
          Ic[j] = j;
		double rho = ((double) rand() / (RAND_MAX)); // (double) otherwise integer division!
		double theta = ((double) rand() / (RAND_MAX));
		p2[j][0] = sqrt(rho)*cos(2*M_PI*theta); p2[j][1] = sqrt(rho)*sin(2*M_PI*theta); p2[j][2] = z2;
		tab2[j]=j;
	}

  // Matrix
	MyMatrix A(p1,p2);

  // Hmatrix
	HMatrix<fullACA,double> HA(A,p1,tab1,p2,tab2);
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
	std::vector<double> x_global(nc,1),f_ref_global(nr),f_global(nr),f_local_to_global(nr);
	f_ref_global=A*x_global;

  // Local vectors
  int nr_local=HA.get_local_size_cluster();
	const std::vector<std::vector<int>>& MasterClusters = HA.get_MasterClusters();
	std::vector<double> x_local(nr_local,1),f_local(nr_local);

	for (int i =0;i<10;i++){
	  // Global mvprod
		MPI_Barrier(HA.get_comm());
		mytime = MPI_Wtime();
	  HA.mvprod_global(x_global.data(),f_global.data());
		MPI_Barrier(HA.get_comm());
		mytime= MPI_Wtime() - mytime;
		MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0,HA.get_comm());
		MPI_Reduce(&mytime, &meantime, 1, MPI_DOUBLE, MPI_SUM, 0,HA.get_comm());
		meantime/=size;
		if (rank==0){
			cout << maxtime<<" "<<meantime<<endl;
		}

	  // Local mvprod
		MPI_Barrier(HA.get_comm());
		mytime = MPI_Wtime();
	  HA.mvprod_local(x_local.data(),f_local.data());
		MPI_Barrier(HA.get_comm());
		mytime= MPI_Wtime() - mytime;
		MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0,HA.get_comm());
		MPI_Reduce(&mytime, &meantime, 1, MPI_DOUBLE, MPI_SUM, 0,HA.get_comm());
		meantime/=size;
		if (rank==0){
			cout << maxtime<<" "<<meantime<<endl;
		}

	  // Local to global
	  std::vector<double> rcv(nr);

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
	      f_local_to_global[MasterClusters[i][j]] = rcv[displs[i]+j];


	  // Errors
	  if (rank==0){
	  	cout << "Error on global mat vec prod : "<<norm2(f_global-f_ref_global)/norm2(f_ref_global)<<std::endl;
	    cout << "Error on local mat vec prod : "<<norm2(f_local_to_global-f_ref_global)/norm2(f_ref_global)<<std::endl;

	  }
	}
	// Finalize the MPI environment.
	MPI_Finalize();
	return test;
}
