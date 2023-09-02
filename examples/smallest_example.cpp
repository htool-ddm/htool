#include <htool/clustering/tree_builder/recursive_build.hpp>
#include <htool/distributed_operator/utility.hpp>
#include <htool/testing/geometry.hpp>

using namespace std;
using namespace htool;

class MyMatrix : public VirtualGenerator<double> {
    const vector<int> &m_target_permutation;
    const vector<int> &m_source_permutation;
    const vector<double> &m_p1;
    const vector<double> &m_p2;
    int m_space_dim;
    int m_nr;
    int m_nc;

  public:
    // Constructor
    MyMatrix(int space_dim0, const vector<int> &target_permutation, const vector<int> &source_permutation, const vector<double> &p10, const vector<double> &p20) : m_target_permutation(target_permutation), m_source_permutation(source_permutation), m_p1(p10), m_p2(p20), m_space_dim(space_dim0), m_nr(target_permutation.size()), m_nc(source_permutation.size()) {}
    double get_coef(const int &k, const int &j) const {
        return 1. / (1 + std::sqrt(std::inner_product(m_p1.begin() + m_space_dim * k, m_p1.begin() + m_space_dim * k + m_space_dim, m_p2.begin() + m_space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }

    // Virtual function to overload
    void copy_submatrix(int M, int N, int row_offset, int col_offset, double *ptr) const override {
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < N; k++) {
                ptr[j + M * k] = this->get_coef(m_target_permutation[j + row_offset], m_source_permutation[k + col_offset]);
            }
        }
    }

    // Matrix vector product
    std::vector<double> operator*(std::vector<double> a) {
        std::vector<double> result(m_nr, 0);
        for (int j = 0; j < m_nr; j++) {
            for (int k = 0; k < m_nc; k++) {
                result[j] += this->get_coef(j, k) * a[k];
            }
        }
        return result;
    }

    // Frobenius norm
    double norm() {
        double norm = 0;
        for (int j = 0; j < m_nr; j++) {
            for (int k = 0; k < m_nc; k++) {
                norm += this->get_coef(j, k);
            }
        }
        return norm;
    }
};

int main(int argc, char *argv[]) {

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    int sizeWorld, rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);

    // Check the number of parameters
    if (argc < 1) {
        // Tell the user how to run the program
        cerr << "Usage: " << argv[0] << " outputpath" << endl;
        /* "Usage messages" are a conventional way of telling the user
         * how to run a program if they enter the command incorrectly.
         */
        return 1;
    }

    std::string outputpath = argv[1];

    // Htool parameters
    double epsilon = 0.001;
    double eta     = 100;

    // Geometry
    int size = 5000;
    vector<double> p(3 * size);
    create_sphere(size, p.data());

    // Clustering
    ClusterTreeBuilder<double> recursive_build_strategy;
    Cluster<double> cluster = recursive_build_strategy.create_cluster_tree(size, 3, p.data(), 2, sizeWorld);

    // Generator
    MyMatrix A(3, cluster.get_permutation(), cluster.get_permutation(), p, p);

    // Distributed operator
    DefaultApproximationBuilder<double, double> default_approximation_builder(A, cluster, cluster, epsilon, eta, 'S', 'U', MPI_COMM_WORLD);
    DistributedOperator<double> &distributed_operator = default_approximation_builder.distributed_operator;

    // Matrix vector product
    std::vector<double> x(size, 1), result(size, 0), ref(size, 0);
    distributed_operator.matrix_product_global_to_global(x.data(), result.data(), 1);
    ref = A * x;
    if (rankWorld == 0) {
        std::cout << "relative error on matrix vector product : ";
        std::cout << norm2(ref - result) / norm2(ref) << "\n";
    }

    // Output
    const HMatrix<double, double> &local_hmatrix = default_approximation_builder.hmatrix;

    if (rankWorld == 0) {
        std::cout << "Tree parameters\n";
        print_tree_parameters(local_hmatrix, std::cout);
        std::cout << "Information about the hmatrix on rank 0\n";
        print_hmatrix_information(local_hmatrix, std::cout);
        std::cout << "Information about hmatrices accross all processors\n";
    }
    print_distributed_hmatrix_information(local_hmatrix, std::cout, MPI_COMM_WORLD);
    save_leaves_with_rank(local_hmatrix, outputpath + "/local_hmatrix_" + std::to_string(rankWorld));

    // Finalize the MPI environment.
    MPI_Finalize();
}
