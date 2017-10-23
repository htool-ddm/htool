#include <htool/htool.hpp>
#include <htool/SVD.hpp>

using namespace std;
using namespace htool;



class MyMatrix: public IMatrix<double>{
	const vector<R3>& p1;
	const vector<R3>& p2;

public:
	MyMatrix(const vector<R3>& p10,const vector<R3>& p20 ):IMatrix<double>(p10.size(),p20.size()),p1(p10),p2(p20) {}
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

int main(int argc, char* argv[]){
	srand (10000);
	// we set a constant seed for rand because we want always the same result if we run the check many times
	// (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

	// Build matrix A with property for ACA
	int nr = 100;
	int nc = 100;
	vector<int> Ir(nr); // row indices for the lrmatrix
	vector<int> Ic(nc); // column indices for the lrmatrix
	// p1: points in a unit ball of the plane z=z1
	double z1 = 0;
	vector<R3> p1(nr);
	vector<int> tab1(nr);
	for(int j=0; j<nr; j++){
		Ir[j] = j;
		double rho = ((double) rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
		double theta = ((double) rand() / (double)(RAND_MAX));
		double phi   = ((double) rand() / (double)(RAND_MAX));

		p1[j][0] = pow(rho,1./3.)*cos(2*M_PI*theta)*sin(M_PI*phi);
		p1[j][1] = pow(rho,1./3.)*sin(2*M_PI*theta)*sin(M_PI*phi);
		p1[j][2] = z1+pow(rho,1./3.)*cos(M_PI*phi);
		tab1[j]=j;

		// sqrt(rho) otherwise the points would be concentrated in the center of the disk
	}

	// p2: points in a unit ball of the plane z=z2
	vector<R3> p2(nc);
	vector<int> tab2(nc);
	double z2=2.;
	for(int j=0; j<nc; j++){

		Ic[j] = j;
		double rho = ((double) rand() / (RAND_MAX)); // (double) otherwise integer division!
		double theta = ((double) rand() / (RAND_MAX));
		double phi   = ((double) rand() / (double)(RAND_MAX));
		p2[j][0] = pow(rho,1./3.)*cos(2*M_PI*theta)*sin(M_PI*phi);
		p2[j][1] = pow(rho,1./3.)*sin(2*M_PI*theta)*sin(M_PI*phi );
		p2[j][2] = z2+pow(rho,1./3.)*cos(M_PI*phi);
		tab2[j]=j;
	}


	// Parametres
	// Load the inputs
	// string inputname = argv[1];
	// LoadParamIO(inputname);
	SetNdofPerElt(1);
// 	string filename=GetOutputPath()+"/output_err_decrease.txt";
// 	ofstream output(filename.c_str());
// //	output.open(filename,ios::app);
// 	if (!output){
// 		cerr<<"Output file cannot be created"<<endl;
// 		exit(1);
// 	}

	double dist=2;
	for (int j=0;j<10;j++){
		dist +=0.5;
		for(int j=0; j<nr; j++){
			p2[j][2] += 0.5;
		}
		Cluster t(p1,tab1); Cluster s(p2,tab2);
		MyMatrix A(p1,p2);

		int reqrank_max = 50;

		partialACA<double> A_partialACA(Ir,Ic,reqrank_max);
		A_partialACA.build(A,t,s);

		SVD<double> A_SVD(Ir,Ic,reqrank_max);
		A_SVD.build(A);

		std::vector<double> partialACA_errors(reqrank_max,0);
		std::vector<double> SVD_errors(reqrank_max,0);

		for (int k = 0 ; k < reqrank_max ; k++){
			partialACA_errors[k]=Frobenius_relative_error(A_partialACA,A,k);
			SVD_errors[k]=Frobenius_relative_error(A_SVD,A,k);
			cout <<dist<<"\t"<<k<<"\t"<<partialACA_errors[k]<<"\t"<<SVD_errors[k]<<endl;
		}
	}
	// output.close();


}
