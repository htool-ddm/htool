#ifndef HTOOL_TESTING_MESH_TEST_HPP
#define HTOOL_TESTING_MESH_TEST_HPP

#include <vector>

namespace htool {

// Small structured mesh for tests: segments (2 points per element) or triangles (3 points per element).
template <int dimension>
struct TestMesh {
    int number_of_points_per_element;
    std::vector<double> points;          // dimension coordinates per point
    std::vector<int> elements_to_points; // number_of_points_per_element point indices per element

    int number_of_points() const { return int(points.size()) / dimension; }
    int number_of_elements() const { return int(elements_to_points.size()) / number_of_points_per_element; }
};

// n x n unit square in the first two axes, each cell split into two triangles, shifted by distance along the last
// axis (z in 3D; y in 2D, where the square has height 1, so two meshes are separated only if distance > 1).
template <int dimension>
TestMesh<dimension> make_triangle_mesh(int n, double distance) {
    static_assert(dimension == 2 || dimension == 3, "triangle meshes are in 2D or 3D");
    TestMesh<dimension> mesh;
    mesh.number_of_points_per_element = 3;
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            mesh.points.push_back(double(i) / n);
            mesh.points.push_back(double(j) / n + (dimension == 2 ? distance : 0.));
            if (dimension == 3)
                mesh.points.push_back(distance);
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int v00 = i * (n + 1) + j, v10 = (i + 1) * (n + 1) + j, v01 = v00 + 1, v11 = v10 + 1;
            mesh.elements_to_points.insert(mesh.elements_to_points.end(), {v00, v10, v01, v10, v11, v01});
        }
    }
    return mesh;
}

// n segments on [0, 1] along x, shifted by distance along the last axis. Points and elements are numbered with a
// stride coprime with n and n + 1 rather than along x, so that a cluster tree built on them does not end up with the
// identity permutation (which would hide user vs tree numbering bugs).
template <int dimension>
TestMesh<dimension> make_segment_mesh(int n, double distance) {
    static_assert(dimension == 2 || dimension == 3, "segment meshes are in 2D or 3D");
    auto gcd = [](int a, int b) {
        while (b != 0) {
            int r = a % b;
            a     = b;
            b     = r;
        }
        return a;
    };
    int stride = 2;
    while (gcd(stride, n) != 1 || gcd(stride, n + 1) != 1)
        stride++;

    TestMesh<dimension> mesh;
    mesh.number_of_points_per_element = 2;
    mesh.points.resize(std::size_t((n + 1) * dimension));
    mesh.elements_to_points.resize(std::size_t(2 * n));
    for (int i = 0; i <= n; i++) {
        std::size_t point = std::size_t((stride * i) % (n + 1));
        for (int d = 0; d < dimension; d++)
            mesh.points[point * std::size_t(dimension) + std::size_t(d)] = d == 0 ? double(i) / n : (d == dimension - 1 ? distance : 0.);
    }
    for (int i = 0; i < n; i++) {
        std::size_t element                      = std::size_t((stride * i) % n);
        mesh.elements_to_points[2 * element]     = (stride * i) % (n + 1);
        mesh.elements_to_points[2 * element + 1] = (stride * (i + 1)) % (n + 1);
    }
    return mesh;
}

} // namespace htool

#endif
