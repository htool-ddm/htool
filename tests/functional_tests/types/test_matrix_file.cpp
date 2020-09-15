#include "htool/types/matrix.hpp"

using namespace std;
using namespace htool;
int main(int argc, char const *argv[]) {
    bool test = 0;

    //// Matrix - double
    Matrix<double> Md(10, 5);
    Matrix<double> Pd(10, 5);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 5; j++) {
            Md(i, j) = i + j;
        }
    }
    test = test || (Md.matrix_to_bytes("Md"));
    test = test || (Pd.bytes_to_matrix("Md"));
    test = test || !(normFrob(Md - Pd) < 1e-16);
    cout << "diff : " << normFrob(Md - Pd) << endl;

    //// Matrix - complex double
    Matrix<complex<double>> Mcd(10, 5);
    Matrix<complex<double>> Pcd(10, 5);

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 5; j++) {
            Mcd(i, j) = complex<double>(i + j, 1);
        }
    }
    test = test || (Mcd.matrix_to_bytes("Mcd"));
    test = test || (Pcd.bytes_to_matrix("Mcd"));
    test = test || !(normFrob(Mcd - Pcd) < 1e-16);
    cout << "diff : " << normFrob(Mcd - Pcd) << endl;

    return test;
}
