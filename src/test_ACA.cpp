#include <iostream>
#include <complex>
#include <vector>
#include <cassert>
#include <htool/htool.hpp>
using namespace std;

/**************************************************************************//**
* Test the precision of the matrix vector product after the ACA compression
* varyng the parameter epsilon of the algorithm and
* the distance between the clusters of points used to generate the considered matrix
*****************************************************************************/

int main(){
	
	const int nepsilon = 4;
	double epsilon[nepsilon];
	epsilon[0] = 1; epsilon[1] = 1e-1; epsilon[2] = 1e-2; epsilon[3] = 1e-3;
	const int ndistance = 4;
	double distance[ndistance];
	distance[0] = 2; distance[1] = 5; distance[2] = 10; distance[3] = 20;
	SetEta(1);
	SetNdofPerElt(1);
	for(int ieps=0; ieps<nepsilon; ieps++)
	{
		// Parametres
		SetEpsilon(epsilon[ieps]); // eta (only for hmatrix), ACA epsilon
		cout << "-> Epsilon of ACA: " << epsilon[ieps] << endl;
		
		for(int idist=0; idist<ndistance; idist++)
		{
			cout << "Distance between the clusters: " << distance[idist] << endl;
			
			srand (1);
			// we set a constant seed for rand because we want always the same result if we run the check many times
			// (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)
			
			// Build matrix A with property for ACA
			int nr = 100;
			vectInt Ir(nr); // row indices for the lrmatrix
			vectInt Ic(nr); // column indices for the lrmatrix
			// p1: points in a unit disk of the plane z=z1
			double z1 = 1;
			vectR3 p1(nr);
			vectReal r1(nr);
			vectInt tab1(nr);
			for(int j=0; j<nr; j++){
				Ir[j] = j;
				Ic[j] = j;
				double rho = ((double) rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
				double theta = ((double) rand() / (double)(RAND_MAX));
				p1[j][0] = sqrt(rho)*cos(2*M_PI*theta); p1[j][1] = sqrt(rho)*sin(2*M_PI*theta); p1[j][2] = z1;
				// sqrt(rho) otherwise the points would be concentrated in the center of the disk
				r1[j]=0.;
				tab1[j]=j;
			}
			// p2: points in a unit disk of the plane z=z2
			double z2 = 1+distance[idist];
			vectR3 p2(nr);
			vectReal r2(nr);
			vectInt tab2(nr);
			for(int j=0; j<nr; j++){
				double rho = ((double) rand() / (RAND_MAX)); // (double) otherwise integer division!
				double theta = ((double) rand() / (RAND_MAX));
				p2[j][0] = sqrt(rho)*cos(2*M_PI*theta); p2[j][1] = sqrt(rho)*sin(2*M_PI*theta); p2[j][2] = z2;
				r2[j]=0.;
				tab2[j]=j;
			}
			Matrix A(nr,nr);
			for(int j=0; j<nr; j++){
				for(int k=0; k<nr; k++){
					A(j,k) = 1./(4*M_PI*norm(p1[j]-p2[k]));
				}
			}
			
			SubMatrix Abis(A,Ir,Ic); // A viewed as a SubMatrix
			Cluster t(p1,r1,tab1); Cluster s(p2,r2,tab2);
			LowRankMatrix B(Abis,Ir,Ic,t,s); // construct a low rank matrix B applying ACA to matrix A
			
			// Vecteur
			vectCplx u(nr);
			int NbSpl = 1000;
			double du = 5./double(NbSpl);
			for(int j=0; j<nr; j++){
				int n = rand()%(NbSpl+1);
				u[j] = n*du;}
			
			vectCplx ua(nr),ub(nr);
			MvProd(ua,A,u);
			MvProd(ub,B,u);
			Real err = norm(ua-ub)/norm(ua);
			cout << "Erreur: " << err << endl;
			
			cout << "Taux de compression: ";
			cout << CompressionRate(B) << endl;
		}
	}
	
}
