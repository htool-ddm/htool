#include <algorithm>                    // for copy
#include <complex>                      // for complex
#include <htool/basic_types/vector.hpp> // for norm2, operator-, bytes_to_v...
#include <iostream>                     // for basic_ostream, operator<<, cout
#include <string>                       // for char_traits, basic_string
#include <vector>                       // for vector

using namespace std;
using namespace htool;
int main(int, char const *[]) {
    bool test = 0;

    //// Vector - double
    vector<double> Vd(10);
    vector<double> Pd(10);
    for (int i = 0; i < 10; i++) {
        Vd[i] = i;
    }
    test = test || (vector_to_bytes(Vd, "Vd"));
    test = test || (bytes_to_vector(Pd, "Vd"));
    test = test || !(norm2(Vd - Pd) < 1e-16);
    cout << "diff : " << norm2(Vd - Pd) << endl;

    //// Vector - complex double
    vector<complex<double>> Vcd(10);
    vector<complex<double>> Pcd(10);

    for (int i = 0; i < 10; i++) {
        Vcd[i] = complex<double>(i, 1);
    }
    test = test || (vector_to_bytes(Vcd, "Vcd"));
    test = test || (bytes_to_vector(Pcd, "Vcd"));
    test = test || !(norm2(Vcd - Pcd) < 1e-16);
    cout << "diff : " << norm2(Vcd - Pcd) << endl;

    cout << test << endl;
    return test;
}
