#include <iostream>
#include <complex>
#include <vector>

#include <htool/htool.hpp>

using namespace std;
using namespace htool;


class MyMatrix: public IMatrix<double>{
	const vector<R3>& p1;
	const vector<R3>& p2;

public:
	MyMatrix(const vector<R3>& p10,const vector<R3>& p20 ):IMatrix<double>(p10.size(),p20.size()),p1(p10),p2(p20) {}
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

    double normFrob (){
        double norm=0;
		for (int i=0;i<p1.size();i++){
			for (int k=0;k<p2.size();k++){
                norm = norm + std::pow(this->get_coef(i,k),2);
            }
        }
        return sqrt(norm);
    }
};


int main(int argc, char* argv[]){

    // Initialize the MPI environment
    MPI_Init(&argc,&argv);

	// Check the number of parameters
	if (argc < 3) {
		// Tell the user how to run the program
		cerr << "Usage: " << argv[0] << " distance \b outputfile \b outputpath" << endl;
		/* "Usage messages" are a conventional way of telling the user
		 * how to run a program if they enter the command incorrectly.
		 */
		return 1;
	}

	double distance = StrToNbr<double>(argv[1]);
	std::string outputfile  = argv[2];
    std::string outputpath  = argv[3];

    SetNdofPerElt(1);
    SetEpsilon(0.0001);
    int reqrank_max = 50;
    srand (1);
    // we set a constant seed for rand because we want always the same result if we run the check many times
    // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

    int nr = 500;
    int nc = 100;
    vector<int> Ir(nr); // row indices for the lrmatrix
    vector<int> Ic(nc); // column indices for the lrmatrix

    double z1 = 1;
    vector<R3>     p1(nr);
    vector<double> r1(nr);
    vector<double> g1(nc,1);
    vector<int>  tab1(nr);
    for(int j=0; j<nr; j++){
        Ir[j] = j;
        double rho = ((double) rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
        double theta = ((double) rand() / (double)(RAND_MAX));
        p1[j][0] = sqrt(rho)*cos(2*M_PI*theta); p1[j][1] = sqrt(rho)*sin(2*M_PI*theta); p1[j][2] = z1;
        // sqrt(rho) otherwise the points would be concentrated in the center of the disk
        r1[j]=0.;
        tab1[j]=j;
    }
    // p2: points in a unit disk of the plane z=z2
    double z2 = 1+distance;
    vector<R3> p2(nc);
    vector<double> r2(nc);
    vector<double> g2(nc,1);
    vector<int> tab2(nc);
    for(int j=0; j<nc; j++){
        Ic[j] = j;
        double rho = ((double) rand() / (RAND_MAX)); // (double) otherwise integer division!
        double theta = ((double) rand() / (RAND_MAX));
        p2[j][0] = sqrt(rho)*cos(2*M_PI*theta); p2[j][1] = sqrt(rho)*sin(2*M_PI*theta); p2[j][2] = z2;
        r2[j]=0.;
        tab2[j]=j;
    }

    // Clustering

    GeometricClustering t, s;
    t.build(p1,r1,tab1,g1);
    s.build(p2,r2,tab2,g2);

    MyMatrix A(p1,p2);
    double norm_A= A.normFrob();

    // SVD with fixed rank
    SVD<double,GeometricClustering> A_SVD(t.get_perm(),s.get_perm(),reqrank_max);
    A_SVD.build(A,t,p1,tab1,s,p2,tab2);
    std::vector<double> SVD_fixed_errors;
    for (int k = 0 ; k < A_SVD.rank_of()+1 ; k++){
        SVD_fixed_errors.push_back(Frobenius_absolute_error(A_SVD,A,k)/norm_A);
    }


    // fullACA with fixed rank
    fullACA<double,GeometricClustering> A_fullACA_fixed(t.get_perm(),s.get_perm(),reqrank_max);
    A_fullACA_fixed.build(A,t,p1,tab1,s,p2,tab2);
    std::vector<double> fullACA_fixed_errors;
    for (int k = 0 ; k < A_fullACA_fixed.rank_of()+1 ; k++){
        fullACA_fixed_errors.push_back(Frobenius_absolute_error(A_fullACA_fixed,A,k)/norm_A);
    }

    // partialACA with fixed rank
    partialACA<double,GeometricClustering> A_partialACA_fixed(t.get_perm(),s.get_perm(),reqrank_max);
    A_partialACA_fixed.build(A,t,p1,tab1,s,p2,tab2);
    std::vector<double> partialACA_fixed_errors;
    for (int k = 0 ; k < A_partialACA_fixed.rank_of()+1 ; k++){
        partialACA_fixed_errors.push_back(Frobenius_absolute_error(A_partialACA_fixed,A,k)/norm_A);
    }

    // sympartialACA with fixed rank
    sympartialACA<double,GeometricClustering> A_sympartialACA_fixed(t.get_perm(),s.get_perm(),reqrank_max);
    A_sympartialACA_fixed.build(A,t,p1,tab1,s,p2,tab2);
    std::vector<double> sympartialACA_fixed_errors;
    for (int k = 0 ; k < A_sympartialACA_fixed.rank_of()+1 ; k++){
        sympartialACA_fixed_errors.push_back(Frobenius_absolute_error(A_sympartialACA_fixed,A,k)/norm_A);
    }

    // Output
    ofstream file_fixed((outputpath+"/"+outputfile).c_str());
    file_fixed<<"Rank,SVD,Full ACA,partial ACA,sym partial ACA"<<endl;
    for (int i=0;i<reqrank_max;i++){
        file_fixed<<i<<","<<SVD_fixed_errors[i]<<","<<fullACA_fixed_errors[i]<<","<<partialACA_fixed_errors[i]<<","<<sympartialACA_fixed_errors[i]<<endl;
    }

    ofstream geometry_1((outputpath+"/geometry_1_"+outputfile).c_str());
    for (int i=0;i<nr;i++){
        geometry_1<<p1[i][0]<<","<<p1[i][1]<<","<<p1[i][2]<<endl;
    }

    ofstream geometry_2((outputpath+"/geometry_2_"+outputfile).c_str());
    for (int i=0;i<nc;i++){
        geometry_2<<p2[i][0]<<","<<p2[i][1]<<","<<p2[i][2]<<endl;
    }

    // Finalize the MPI environment.
    MPI_Finalize();
}
