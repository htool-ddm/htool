#include <iostream>
#include <complex>
#include <vector>
#include <random>

#include <htool/multilrmat/multilrmat.hpp>
#include <htool/clustering/ncluster.hpp>


using namespace std;
using namespace htool;

class MyMultiMatrix: public MultiIMatrix<double>{
	const vector<R3>& p1;
	const vector<R3>& p2;

public:
	MyMultiMatrix(const vector<R3>& p10,const vector<R3>& p20 ):MultiIMatrix<double>(p10.size(),p20.size(),5),p1(p10),p2(p20) {}

	std::vector<double> get_coefs(const int& i, const int& j)const {
		return std::vector<double> {
			1./(4*M_PI*norm2(p1[i]-p2[j])),
			2./(4*M_PI*norm2(p1[i]-p2[j])),
			3./(4*M_PI*norm2(p1[i]-p2[j])),
			4./(4*M_PI*norm2(p1[i]-p2[j])),
			5./(4*M_PI*norm2(p1[i]-p2[j]))
		};
	}
	
	std::vector<double> mult(std::vector<double>& a, int l) const{
		std::vector<double> result(p1.size(),0);
		for (int i=0;i<p1.size();i++){
			for (int k=0;k<p2.size();k++){
				result[i]+=this->get_coefs(i,k)[l]*a[k];
			}
		}
		return result;
	 }
};

class MyMatrix: public IMatrix<double>{
	const vector<R3>& p1;
	const vector<R3>& p2;

public:
	MyMatrix(const vector<R3>& p10,const vector<R3>& p20 ):IMatrix<double>(p10.size(),p20.size()),p1(p10),p2(p20) {}

	double get_coef(const int& i, const int& j)const {
		return 1./(4*M_PI*norm2(p1[i]-p2[j]));
	}
	
	std::vector<double> mult(std::vector<double>& a) const{
		std::vector<double> result(p1.size(),0);
		for (int i=0;i<p1.size();i++){
			for (int k=0;k<p2.size();k++){
				result[i]+=this->get_coef(i,k)*a[k];
			}
		}
		return result;
	 }
};

void create_geometry(int distance, std::vector<R3>& xt, std::vector<int>& tabt, std::vector<R3>& xs, std::vector<int>& tabs){
	cout << "Distance between the clusters: " << NbrToStr(distance) << endl;
	
	srand (1);
	// we set a constant seed for rand because we want always the same result if we run the check many times
	// (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

	int nr = xt.size();
	int nc = xs.size();
	vector<int> Ir(nr); // row indices for the lrmatrix
	vector<int> Ic(nc); // column indices for the lrmatrix

	double z1 = 1;
	for(int j=0; j<nr; j++){
		Ir[j] = j;
		double rho = ((double) rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
		double theta = ((double) rand() / (double)(RAND_MAX));
		xt[j][0] = sqrt(rho)*cos(2*M_PI*theta); xt[j][1] = sqrt(rho)*sin(2*M_PI*theta); xt[j][2] = z1;
		// sqrt(rho) otherwise the points would be concentrated in the center of the disk
		tabt[j]=j;
	}
	// p2: points in a unit disk of the plane z=z2
	double z2 = 1+distance;
	vector<int> tab2(nc);
	for(int j=0; j<nc; j++){
		Ic[j] = j;
		double rho = ((double) rand() / (RAND_MAX)); // (double) otherwise integer division!
		double theta = ((double) rand() / (RAND_MAX));
		xs[j][0] = sqrt(rho)*cos(2*M_PI*theta); xs[j][1] = sqrt(rho)*sin(2*M_PI*theta); xs[j][2] = z2;
		tabs[j]=j;
	}
}

template< class MultiLowRankMatrix >
int test_multi_lrmat(const MyMultiMatrix& A,const MultiLowRankMatrix& Fixed_approximation, const MultiLowRankMatrix& Auto_approximation, const std::vector<int>& permt, const std::vector<int>& perms, std::pair<double,double> fixed_compression_interval, std::pair<double,double> auto_compression_interval){

	bool test = 0;
	int nr=permt.size();
	int nc=perms.size();
	
	// Random vector
	double lower_bound = 0;
	double upper_bound = 10000;
	std::random_device rd;
	std::mt19937 mersenne_engine(rd());
	std::uniform_real_distribution<double> dist(lower_bound,upper_bound);
	auto gen = [&dist, &mersenne_engine](){
				return dist(mersenne_engine);
			};

	vector<double> f(nc,1);
	generate(begin(f), end(f), gen);

	// ACA with fixed rank
	int reqrank_max = 10;
	std::vector<std::vector<double>> fixed_errors;
	for (int k = 0 ; k < reqrank_max+1 ; k++){
		fixed_errors.push_back(Frobenius_absolute_error(Fixed_approximation,A,k));
	}


	// Test rank
	test = test || !(Fixed_approximation.rank_of()==reqrank_max);
	std::cout << test << " "  << reqrank_max<< " "<<Fixed_approximation.rank_of()<< std::endl;
	cout << "Compression with fixed rank" << endl;
	cout << "> rank : "<<Fixed_approximation.rank_of() << endl;

	// Test Frobenius errors
	test = test || !(max(fixed_errors.back())<1e-7);
	cout << "> Errors with Frobenius norm : "<<fixed_errors<<endl;

	for (int l=0;l<A.nb_matrix();l++){

		// Test compression
		test = test || !(fixed_compression_interval.first<Fixed_approximation[l].compression() && Fixed_approximation[l].compression()<fixed_compression_interval.second);
		cout << "> Compression rate : "<<Fixed_approximation[l].compression()<<endl;


		// Test mat vec prod
		std::vector<double> out_perm(nr);
		std::vector<double> out=Fixed_approximation[l]*f;
		for (int i = 0; i<permt.size();i++){
			out_perm[permt[i]]=out[i];
		}
		double error=norm2(A.mult(f,l)-out_perm)/norm2(A.mult(f,l));
		test = test || !(error<GetEpsilon()*10);
		cout << "> Errors on a mat vec prod : "<< error<< " " << (GetEpsilon()*10)<<" "<<(error<GetEpsilon()*10)<<endl;
		cout << "test : "<<test<<endl<<endl;

	}


	// ACA automatic building
	std::vector<std::vector<double>> auto_errors;
	for (int k = 0 ; k < Auto_approximation.rank_of()+1 ; k++){
		auto_errors.push_back(Frobenius_absolute_error(Auto_approximation,A,k));
	}
	cout << "Automatic compression" << endl;

	// Test Frobenius error
	test = test || !(max(auto_errors[Auto_approximation.rank_of()])<GetEpsilon());
	cout << "> Errors with Frobenius norm: "<<auto_errors<<endl;

	for (int l=0;l<A.nb_matrix();l++){
		
		// Test compression rate
		test = test || !(auto_compression_interval.first<Auto_approximation[l].compression() && Auto_approximation[l].compression()<auto_compression_interval.second);
		cout << "> Compression rate : "<<Auto_approximation[l].compression()<<endl;

		// Test mat vec prod
		std::vector<double> out_perm(nr);
		std::vector<double> out=Auto_approximation[l]*f;
		for (int i = 0; i<permt.size();i++){
			out_perm[permt[i]]=out[i];
		}
		double error = norm2(A.mult(f,l)-out_perm)/norm2(A.mult(f,l));
		test = test || !(error<GetEpsilon()*10);
		cout << "> Errors on a mat vec prod : "<< error<<endl;


		cout << "test : "<<test<<endl<<endl;
	}

	return test;
}
