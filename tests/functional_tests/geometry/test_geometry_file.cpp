#include <array>
#include <htool/geometry.hpp>
#include <htool/point.hpp>
#include <htool/vector.hpp>
using namespace std;
using namespace htool;

int main(int argc, char const *argv[]) {

    // Input file
    if (argc != 2) { // argc should be 2 for correct execution
        // We print argv[0] assuming it is the program name
        cout << "usage: " << argv[0] << " <meshfile>\n";
        return 1;
    }

    bool test = 0;
    std::vector<R3> points;
    std::vector<double> r;

    test = test || (Load_GMSH_nodes(points, argv[1]));
    test = test || !(norm2(points[0] - R3{1, 0, 0}) < 1e-16);
    test = test || !(norm2(points[1] - R3{0, 1, 0}) < 1e-16);
    test = test || !(norm2(points[2] - R3{-1, 0, 0}) < 1e-16);
    test = test || !(norm2(points[3] - R3{0, -1, 0}) < 1e-16);

    cout << test << endl;
    return test;
}
