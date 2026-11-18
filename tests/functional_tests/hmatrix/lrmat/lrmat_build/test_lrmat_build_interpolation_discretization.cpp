#include <htool/clustering/cluster_node.hpp>
#include <htool/hmatrix/lrmat/interpolation/fem_discretization.hpp>
#include <htool/hmatrix/lrmat/interpolation/point_discretization.hpp>
#include <htool/matrix/linalg/add_matrix_matrix_product.hpp>
#include <htool/matrix/utils.hpp>
#include <htool/testing/fem_test.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace htool;

template <int dimension>
bool is_inside(const interpolation_box<double, dimension> &box, const double *point) {
    for (int d = 0; d < dimension; d++) {
        if (point[d] < box.first[d] || point[d] > box.second[d])
            return false;
    }
    return true;
}

// For each tested dof range and order: the box covers the given points, and the dense interpolation matrix
// equals the tensor form to_dense(p2m) * W.
template <typename CoefficientPrecision, int dimension>
bool check_discretization(const VirtualDiscretization<CoefficientPrecision, double, dimension> &discretization, int number_of_dofs, const std::function<std::vector<const double *>(int, int)> &points_to_cover, const string &name) {
    bool is_error = false;
    for (auto range : {std::make_pair(0, number_of_dofs), std::make_pair(number_of_dofs / 3, number_of_dofs / 2)}) {
        int offset = range.first, size = range.second;
        for (int L : {3, 5}) {
            int n_interp = 1;
            for (int d = 0; d < dimension; d++)
                n_interp *= L;
            auto box = discretization.box(offset, size);
            for (const double *point : points_to_cover(offset, size)) {
                is_error = is_error || !is_inside<dimension>(box, point);
            }

            Matrix<CoefficientPrecision> dense = discretization.interpolation_matrix(L, box, offset, size);
            theia::TensorP2M<dimension, CoefficientPrecision> p2m;
            std::vector<interpolation_weight<CoefficientPrecision>> w;
            discretization.tensor_interpolation(L, box, offset, size, p2m, w);

            Matrix<CoefficientPrecision> p2m_dense(n_interp, p2m.cols()), W(p2m.cols(), size), tensor(n_interp, size);
            CoefficientPrecision *p2m_data = p2m_dense.data();
            p2m.to_dense(p2m_data);
            for (const auto &entry : w) {
                W(entry.particle, entry.dof) += entry.value;
            }
            add_matrix_matrix_product('N', 'N', CoefficientPrecision(1), p2m_dense, W, CoefficientPrecision(0), tensor);

            double difference = normFrob(dense - tensor) / normFrob(dense);
            cout << "  " << name << " range [" << offset << ", " << offset + size << ") L=" << L << ": dense " << dense.nb_rows() << "x" << dense.nb_cols() << ", dense vs tensor " << difference << '\n';
            is_error = is_error || dense.nb_rows() != n_interp || dense.nb_cols() != size || !(difference < 1e-12);
        }
    }
    if (is_error)
        cout << "  FAILED: " << name << '\n';
    return is_error;
}

// Points: the P0 dof points (element centroids) of a mesh.
template <typename CoefficientPrecision, int dimension>
bool test_point_discretization(const TestMesh<dimension> &mesh, const string &name) {
    TestDofs dofs = make_dofs(mesh, ElementType::P0);
    std::vector<double> center(dimension, 0);
    Cluster<double> cluster(0, center, 0, 0, dofs.number_of_dofs()); // single node, identity permutation
    PointDiscretization<CoefficientPrecision, double, dimension> discretization(dofs.dof_points.data(), cluster);

    auto points_to_cover = [&](int offset, int size) {
        std::vector<const double *> points;
        for (int i = 0; i < size; i++)
            points.push_back(dofs.dof_points.data() + dimension * cluster.get_permutation()[offset + i]);
        return points;
    };
    return check_discretization<CoefficientPrecision, dimension>(discretization, dofs.number_of_dofs(), points_to_cover, name);
}

// FEM: the box must cover the vertices of every element incident to the range's dofs.
template <typename CoefficientPrecision, int dimension>
bool test_fem_discretization(const TestMesh<dimension> &mesh, ElementType element, const string &name) {
    TestDofs dofs = make_dofs(mesh, element);
    std::vector<double> center(dimension, 0);
    Cluster<double> cluster(0, center, 0, 0, dofs.number_of_dofs()); // single node, identity permutation
    FEMDiscretization<CoefficientPrecision, double, dimension> discretization(make_fem_space<CoefficientPrecision, dimension>(mesh, dofs, element, 10), cluster);

    auto points_to_cover = [&](int offset, int size) {
        std::vector<const double *> points;
        for (int i = 0; i < size; i++) {
            const std::vector<int> &elements = dofs.dofs_to_elements.at(cluster.get_permutation()[offset + i]);
            for (std::size_t l = 0; l < elements.size(); l += 2) {
                for (int p = 0; p < mesh.number_of_points_per_element; p++)
                    points.push_back(mesh.points.data() + dimension * mesh.elements_to_points[mesh.number_of_points_per_element * elements[l] + p]);
            }
        }
        return points;
    };
    return check_discretization<CoefficientPrecision, dimension>(discretization, dofs.number_of_dofs(), points_to_cover, name);
}

int main(int, char *[]) {
    using complex_type       = std::complex<double>;
    bool is_error            = false;
    TestMesh<3> triangles    = make_triangle_mesh<3>(4, 0);
    TestMesh<2> triangles_2d = make_triangle_mesh<2>(4, 0);
    TestMesh<2> segments_2d  = make_segment_mesh<2>(8, 0);
    is_error                 = test_point_discretization<double, 3>(triangles, "points") || is_error;
    is_error                 = test_point_discretization<complex_type, 3>(triangles, "points complex") || is_error;
    is_error                 = test_fem_discretization<double, 3>(triangles, ElementType::P0, "FEM P0") || is_error;
    is_error                 = test_fem_discretization<double, 3>(triangles, ElementType::P1, "FEM P1") || is_error;
    is_error                 = test_fem_discretization<complex_type, 3>(triangles, ElementType::P1, "FEM P1 complex") || is_error;
    is_error                 = test_fem_discretization<double, 2>(triangles_2d, ElementType::P1, "FEM P1 triangles 2D") || is_error;
    is_error                 = test_fem_discretization<double, 2>(segments_2d, ElementType::P1, "FEM P1 segments 2D") || is_error;
    return is_error;
}
