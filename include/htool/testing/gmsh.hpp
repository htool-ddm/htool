#ifndef HTOOL_GEOMETRY_HPP
#define HTOOL_GEOMETRY_HPP

#include "point.hpp" // for operator>>, R3
#include <algorithm> // for copy, max
#include <array>     // for array
#include <fstream>   // for basic_ifstream, basic_istream::operator>>, basi...
#include <iostream>  // for cout
#include <sstream>   // for basic_istringstream
#include <string>    // for getline, basic_string, operator!=, char_traits
#include <vector>    // for vector

namespace htool {

int Load_GMSH_nodes(std::vector<R3> &x, const std::string &filename) {

    int size = 0;

    std::istringstream iss;
    std::ifstream file;
    std::string line;

    // Open file
    file.open(filename);
    if (!file.good()) {
        std::cout << "Cannot open mesh file\n"; // LCOV_EXCL_LINE
        return 1;                               // LCOV_EXCL_LINE
    }

    // Number of elements
    while (line != "$Nodes") {
        getline(file, line);
    }
    file >> size;
    getline(file, line);
    x.resize(size);

    // Read point
    R3 coord;
    int dummy;
    getline(file, line);
    for (int p = 0; p < size; p++) {
        iss.str(line);
        iss >> dummy;
        iss >> coord;
        x[p] = coord;
        iss.clear();
        getline(file, line);
    }

    // Fermeture fichier
    file.close();

    return 0;
}

} // namespace htool

#endif
