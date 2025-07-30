#include <complex>                      // for complex
#include <htool/matrix/matrix.hpp>      // for Matrix
#include <htool/matrix/matrix_view.hpp> // for MatrixView
#include <htool/matrix/utils.hpp>       // for normFrob
#include <iostream>                     // for basic_ostream, operator<<, cout
#include <string>                       // for basic_string, char_traits

using namespace std;
using namespace htool;
int main(int, char const *[]) {
    bool test = 0;

    //// Matrix - double
    Matrix<double> Md(10, 5);
    Matrix<double> Pd(10, 5);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 5; j++) {
            Md(i, j) = i + j;
        }
    }
    matrix_to_bytes(Md, "Md");
    bytes_to_matrix("Md", Pd);
    test = test || !(normFrob(Md - Pd) < 1e-16);
    cout << "diff : " << normFrob(Md - Pd) << endl;

    //// Matrix view - double
    MatrixView<const double> Md_view(Md);
    matrix_to_bytes(Md_view, "Md_view");
    bytes_to_matrix("Md_view", Pd);
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
    matrix_to_bytes(Mcd, "Mcd");
    bytes_to_matrix("Mcd", Pcd);
    test = test || !(normFrob(Mcd - Pcd) < 1e-16);
    cout << "diff : " << normFrob(Mcd - Pcd) << endl;

    return test;
}
