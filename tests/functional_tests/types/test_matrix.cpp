#include "htool/testing/generator_test.hpp"
#include "htool/types/matrix.hpp"

using namespace std;
using namespace htool;
int main(int argc, char const *argv[]) {
    bool test = 0;

    //// Matrix - double
    // Constructor
    Matrix<double> Md(10, 5);
    Matrix<double> Pd(5, 10);
    double error = normFrob(Md) + Md.nb_rows() - 10 + Md.nb_cols() - 5;
    test         = test || !(error < 1e-16);
    cout << "Error in empty constructor = " << error << endl;

    // Access operator
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 5; j++) {
            Md(i, j) = i + j;
            Pd(j, i) = 2;
        }
    }
    // Assignement operator
    Matrix<double> Nd = Md;
    error             = normFrob(Nd - Md);
    test              = test || !(error < 1e-16);
    cout << "Error on assignement operator : " << error << endl;

    // Getters for strides
    vector<double> diff = {1, 2, 3, 4, 5};
    error               = norm2(Md.get_row(1) - diff);
    test                = test || !(error < 1e-16);
    cout << "Error on row getter : " << error << endl;
    diff  = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    error = norm2(Md.get_col(3) - diff);
    test  = test || !(error < 1e-16);
    cout << "Error on col getter : : " << error << endl;

    // Setters for strides
    std::vector<double> rowd(5, 2);
    std::vector<double> cold(10, 2);
    Md.set_row(1, rowd);
    diff  = {2, 2, 2, 2, 2};
    error = norm2(Md.get_row(1) - diff);
    test  = test || !(error < 1e-16);
    Md.set_col(4, cold);
    diff  = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    error = norm2(Md.get_col(4) - diff);
    test  = test || !(error < 1e-16);
    cout << "Error on row and col setters : " << error << endl;

    // Matrix scalar product
    Matrix<double> Md_test = 2 * Md;
    error                  = 0;
    for (int i = 0; i < Md.nb_rows(); i++) {
        for (int j = 0; j < Md.nb_cols(); j++) {
            error += pow(2 * Md(i, j) - Md_test(i, j), 2);
        }
    }
    error = sqrt(error);
    test  = test || !(error < 1e-16);
    cout << "Error on mat scal prod : " << error << endl;
    error = normFrob(Md * 2 - 2 * Md);
    test  = test || !(error < 1e-16);
    cout << "Error on mat scal prod associativity : " << error << endl;

    // Matrix vector product
    std::vector<double> md(5, 1);
    diff  = {8, 10, 16, 20, 24, 28, 32, 36, 40, 44};
    error = norm2(Md * md - diff);
    test  = test || !(error < 1e-16);
    cout << "Error on mat vec prod : " << error << endl;

    // Matrix matrix product
    Matrix<double> MMd = Md * Pd;
    Matrix<double> MMd_test(Md.nb_rows(), Pd.nb_cols());

    for (int i = 0; i < Md.nb_rows(); i++) {
        for (int j = 0; j < Pd.nb_cols(); j++) {
            for (int k = 0; k < Pd.nb_rows(); k++) {
                MMd_test(i, j) += Md(i, k) * Pd(k, j);
            }
        }
    }
    error = normFrob(MMd - MMd_test);
    test  = test || !(error < 1e-16);
    cout << "Error on matrix matrix product : " << error << endl;

    // Matrix argmax
    test = test || !(argmax(Md).first - 9 == 0);
    test = test || !(argmax(Md).second - 3 == 0);
    cout << "Md's argmax : " << argmax(Md).first << " " << argmax(Md).second << endl;
    cout << "Expected Argmax : (9,3)" << endl;

    //// Matrix - complex double
    // Constructor
    Matrix<complex<double>> Mcd(10, 5);
    Matrix<complex<double>> Pcd(5, 10);
    error = normFrob(Mcd) + Mcd.nb_rows() - 10 + Mcd.nb_cols() - 5;
    test  = test || !(error < 1e-16);
    cout << "Error in empty constructor = " << error << endl;
    // Access operator
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 5; j++) {
            Mcd(i, j) = i + j;
            Pcd(j, i) = 2;
        }
    }
    // Assignement operator
    Matrix<complex<double>> Ncd = Mcd;
    error                       = normFrob(Ncd - Mcd);
    test                        = test || !(error < 1e-16);

    // Getters for strides
    vector<complex<double>> diffc = {1, 2, 3, 4, 5};
    error                         = norm2(Mcd.get_row(1) - diffc);
    test                          = test || !(error < 1e-16);
    cout << "Error on row getter : " << error << endl;
    diffc = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    error = norm2(Mcd.get_col(3) - diffc);
    test  = test || !(error < 1e-16);
    cout << "Error on col getter : " << error << endl;

    // Setters for strides
    std::vector<std::complex<double>> rowcd(5, 2);
    std::vector<std::complex<double>> colcd(10, 2);
    Mcd.set_row(1, rowcd);
    diffc = {2, 2, 2, 2, 2};
    error = norm2(Mcd.get_row(1) - diffc);
    test  = test || !(error < 1e-16);
    Mcd.set_col(4, colcd);
    diffc = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    error = norm2(Mcd.get_col(4) - diffc);
    test  = test || !(error < 1e-16);
    cout << "Error on row and col setters : " << error << endl;

    // Matrix scalar product
    Matrix<std::complex<double>> Mcd_test = 2 * Mcd;
    error                                 = 0;
    for (int i = 0; i < Md.nb_rows(); i++) {
        for (int j = 0; j < Md.nb_cols(); j++) {
            error += pow(abs(2 * Md(i, j) - Md_test(i, j)), 2);
        }
    }
    error = sqrt(error);
    test  = test || !(error < 1e-16);
    cout << "Error on mat scal prod : " << error << endl;
    error = normFrob(Md * 2 - 2 * Md);
    test  = test || !(error < 1e-16);
    cout << "Error on mat scal prod associativity : " << error << endl;

    // Matrix vector product
    std::vector<complex<double>> mcd(5, 1);
    diffc = {8, 10, 16, 20, 24, 28, 32, 36, 40, 44};
    error = norm2(Mcd * mcd - diffc);
    test  = test || !(error < 1e-16);
    cout << "Error matrix vector product : " << error << endl;

    // Matrix matrix product
    Matrix<complex<double>> MMcd = Mcd * Pcd;
    Matrix<complex<double>> MMcd_test(Mcd.nb_rows(), Pcd.nb_cols());

    for (int i = 0; i < Mcd.nb_rows(); i++) {
        for (int j = 0; j < Pcd.nb_cols(); j++) {
            for (int k = 0; k < Pcd.nb_rows(); k++) {
                MMcd_test(i, j) += Mcd(i, k) * Pcd(k, j);
            }
        }
    }
    error = normFrob(MMcd - MMcd_test);
    test  = test || !(error < 1e-16);
    cout << "Error on matrix matrix product : " << error << endl;

    // Matrix argmax
    test = test || !(argmax(Mcd).first - 9 == 0);
    test = test || !(argmax(Mcd).second - 3 == 0);
    cout << "Mcd's argmax : " << argmax(Mcd).first << " " << argmax(Mcd).second << endl;
    cout << "Expected Argmax : (9,3)" << endl;

    cout << test << endl;
    return test;
}
