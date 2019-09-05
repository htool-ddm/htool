#include <htool/htool.hpp>

using namespace std;
using namespace htool;

class MyMatrix: public IMatrix<double>{
    const vector<R3>& p1;
    const vector<R3>& p2;

public:
    // Constructor
    MyMatrix(const vector<R3>& p10,const vector<R3>& p20 ):
        IMatrix(p10.size(),p20.size()),p1(p10),p2(p20) {}

    // Virtual function to overload
    double get_coef(const int& i, const int& j)const {
        R3 dist; 
        dist[0] = p1[i][0]-p2[j][0];
        dist[1] = p1[i][1]-p2[j][1];
        dist[2] = p1[i][2]-p2[j][2];
        return 1./(norm2(dist));
    }

    // Matrix vector product
	std::vector<double> operator*(std::vector<double> a){
		std::vector<double> result(p1.size(),0);
		for (int j=0;j<p1.size();j++){
			for (int k=0;k<p2.size();k++){
				result[j]+=this->get_coef(j,k)*a[k];
			}
		}
        return result;
    }

    // Frobenius norm
    double norm(){
        double norm = 0;
        for (int j=0;j<p1.size();j++){
            for (int k=0;k<p2.size();k++){
                norm+=this->get_coef(j,k);
            }
        }
        return norm;
    }
};


int main(int argc, char *argv[]) {

    // Initialize the MPI environment
    MPI_Init(&argc,&argv);

    // Htool parameters
	SetEpsilon(0.001);
	SetEta(100);

    // Data
    srand (1);    
    int nr = 10000;
    int nc = 5000;
    vector<int> Ir(nr); // row indices for the hmatrix
    vector<int> Ic(nc); // column indices for the hmatrix

    // p1: points in a unit disk of the plane z=z1
    double z1 = 1;
    vector<R3> p1(nr);
    for(int j=0; j<nr; j++){
        Ir[j] = j;
        double rho = ((double) rand() / (double)(RAND_MAX));
        double theta = ((double) rand() / (double)(RAND_MAX));
        p1[j][0] = sqrt(rho)*cos(2*M_PI*theta); 
        p1[j][1] = sqrt(rho)*sin(2*M_PI*theta); 
        p1[j][2] = z1;
    }

    // p2: points in a unit disk of the plane z=z2
    double z2 = 2;
    vector<R3> p2(nc);
    for(int j=0; j<nc; j++){
        Ic[j] = j;
        double rho = ((double) rand() / (RAND_MAX));
        double theta = ((double) rand() / (RAND_MAX));
        p2[j][0] = sqrt(rho)*cos(2*M_PI*theta); 
        p2[j][1] = sqrt(rho)*sin(2*M_PI*theta); 
        p2[j][2] = z2;
    }

    // Hmatrix
    MyMatrix A(p1,p2);
    std::vector<double> x(nc,1),result(nr,0);
    HMatrix<fullACA,double> HA(A,p1,p2);
    result = HA*x;

    // Output
    HA.print_infos();
    std::cout<< Frobenius_absolute_error(HA,A)/A.norm()<<std::endl;
    std::cout<< norm2(A*x-result)/norm2(A*x)<<std::endl;

    // Finalize the MPI environment.
    MPI_Finalize();
}