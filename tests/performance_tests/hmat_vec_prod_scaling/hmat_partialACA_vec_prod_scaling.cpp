#include <htool/partialACA.hpp>
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
	for(int j=0; j<nr; j++){
		Ir[j] = j;
		double rho = ((double) rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
		double theta = ((double) rand() / (double)(RAND_MAX));
		p1[j][0] = sqrt(rho)*cos(2*M_PI*theta); p1[j][1] = sqrt(rho)*sin(2*M_PI*theta); p1[j][2] = z1;
		// sqrt(rho) otherwise the points would be concentrated in the center of the disk
	}
	// p2: points in a unit disk of the plane z=z2
	double z2 = 1+distance;
	vector<R3> p2(nc);
	for(int j=0; j<nc; j++){
          Ic[j] = j;
		double rho = ((double) rand() / (RAND_MAX)); // (double) otherwise integer division!
		double theta = ((double) rand() / (RAND_MAX));
		p2[j][0] = sqrt(rho)*cos(2*M_PI*theta); p2[j][1] = sqrt(rho)*sin(2*M_PI*theta); p2[j][2] = z2;
	}

  // Matrix
	MyMatrix A(p1,p2);

  // Hmatrix
	HMatrix<partialACA,double> HA(A,p1,p2);

	HA.print_stats();
	double mytime, maxtime, meantime;
	double meanmax, meanmean;

  // Global vectors
	std::vector<double> x_global(nc,1),f_global(nr);


	// Global products
	for (int i =0;i<10;i++){
		MPI_Barrier(HA.get_comm());
		mytime = MPI_Wtime();
		f_global=HA*x_global;
		mytime= MPI_Wtime() - mytime;
		MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0,HA.get_comm());
		MPI_Reduce(&mytime, &meantime, 1, MPI_DOUBLE, MPI_SUM, 0,HA.get_comm());
		meantime/=size;
		if (i>4){
			meanmean += meantime;
			meanmax  += maxtime;
		}
	}
	meanmax /= 5;
	meanmean /= 5;
	if (rank==0){
		std::cout <<"Five mvprod: "<<"max ="<<meanmax<<"\t"<<"mean = "<<meanmean<<std::endl;
	}

	// Finalize the MPI environment.
	MPI_Finalize();
	return 0;
}
