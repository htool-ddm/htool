#include <htool/htool.hpp>

using namespace std;
using namespace htool;


class MyMatrix: public IMatrix<double>{
	const vector<R3>& p1;
	const vector<R3>& p2;

public:
	MyMatrix(const vector<R3>& p10,const vector<R3>& p20 ):IMatrix(p10.size(),p20.size()),p1(p10),p2(p20) {}

	double get_coef(const int& i, const int& j)const {return 1./(norm2(p1[i]-p2[j]));}


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

template<typename ClusterImpl, template<typename,typename> class LowRankMatrix>
int hmat(int argc, char *argv[]){

	// Initialize the MPI environment
	MPI_Init(&argc,&argv);

	// Get the number of processes
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Get the rank of the process
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Check the number of parameters
	if (argc < 3) {
		// Tell the user how to run the program
		cerr << "Usage: " << argv[0] << " distance \b outputfile \b outputpath \b epsilon \b eta \b minclustersize \b nr \b nc" << endl;
		/* "Usage messages" are a conventional way of telling the user
		 * how to run a program if they enter the command incorrectly.
		 */
		return 1;
	}

	double distance = StrToNbr<double>(argv[1]);
	std::string outputfile  = argv[2];
    std::string outputpath  = argv[3];
	double epsilon = StrToNbr<double>(argv[4]);
	double eta = StrToNbr<double>(argv[5]);
	double minclustersize = StrToNbr<double>(argv[6]);
	int nr = StrToNbr<int>(argv[7]);
	int nc = StrToNbr<int>(argv[8]);

	//
	SetEpsilon(epsilon);
	SetEta(eta);
	SetMinClusterSize(minclustersize);

  // Create points randomly
	srand (1);
	// we set a constant seed for rand because we want always the same result if we run the check many times
	// (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

	vector<int> Ir(nr); // row indices for the lrmatrix
	vector<int> Ic(nc); // column indices for the lrmatrix

	double z1 = 1;
	vector<R3>     p1(nr);
	vector<double> r1(nr,0);
	vector<double> g1(nr,1);
	vector<int>    tab1(nr);
	for(int j=0; j<nr; j++){
		Ir[j] = j;
		double rho = ((double) rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
		double theta = ((double) rand() / (double)(RAND_MAX));
		p1[j][0] = sqrt(rho)*cos(2*M_PI*theta); p1[j][1] = sqrt(rho)*sin(2*M_PI*theta); p1[j][2] = z1;
		tab1[j]=j;
		// sqrt(rho) otherwise the points would be concentrated in the center of the disk
	}
	// p2: points in a unit disk of the plane z=z2
	double z2 = 1+distance;
	vector<R3> p2(nc);
	vector<double> r2(nc,0);
	vector<double> g2(nc,1);
	vector<int>    tab2(nc);
	for(int j=0; j<nc; j++){
			Ic[j] = j;
		double rho = ((double) rand() / (RAND_MAX)); // (double) otherwise integer division!
		double theta = ((double) rand() / (RAND_MAX));
		p2[j][0] = sqrt(rho)*cos(2*M_PI*theta); p2[j][1] = sqrt(rho)*sin(2*M_PI*theta); p2[j][2] = z2;
		tab2[j]=j;
	}

  // Matrix
	MyMatrix A(p1,p2);

	// Clustering
	std::shared_ptr<ClusterImpl>  t=make_shared<ClusterImpl>();
	std::shared_ptr<ClusterImpl> s=make_shared<ClusterImpl>();
	t->build(p1,r1,tab1,g1,2);
	s->build(p2,r2,tab2,g2,2); 
  // Hmatrix
	HMatrix<double,LowRankMatrix,ClusterImpl> HA(A,p1,p2);

	
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

	std::ofstream output;
	if (rank==0){
		output.open((outputpath+"/"+outputfile).c_str());
		output<<"# Hmatrix"<<std::endl;
	}
	HA.add_info("Mean_global_mat_vec_prod",NbrToStr(meanmean));
	HA.add_info("Max_global_mat_vec_prod",NbrToStr(meanmax));
	HA.save_infos((outputpath+"/"+outputfile).c_str(),std::ios::app,": ");

	// Finalize the MPI environment.
	MPI_Finalize();
	return 0;
}
