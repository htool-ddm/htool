#include <complex>
#include <iostream>
#include <vector>

#include <htool/htool.hpp>
#include <htool/testing/geometry.hpp>
using namespace std;
using namespace htool;

class MyMatrix : public VirtualInternalGenerator<double> {
    const vector<int> &m_target_permutation;
    const vector<int> &m_source_permutation;
    const vector<double> &m_target_coordinates;
    const vector<double> &m_source_coordinates;
    int m_space_dim;
    int m_nr;
    int m_nc;

  public:
    // Constructor
    MyMatrix(int spatial_dimension, const vector<int> &target_permutation, const vector<int> &source_permutation, const vector<double> &target_coordinates, const vector<double> &source_coordinates) : m_target_permutation(target_permutation), m_source_permutation(source_permutation), m_target_coordinates(target_coordinates), m_source_coordinates(source_coordinates), m_space_dim(spatial_dimension), m_nr(target_permutation.size()), m_nc(source_permutation.size()) {}

    // Virtual function to overload, necessary
    void copy_submatrix(int M, int N, int row_offset, int col_offset, double *ptr) const override {
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < N; k++) {
                ptr[j + M * k] = this->get_coef(m_target_permutation[j + row_offset], m_source_permutation[k + col_offset]);
            }
        }
    }

    double get_coef(const int &k, const int &j) const {
        return 1. / (1 + std::sqrt(std::inner_product(m_target_coordinates.begin() + m_space_dim * k, m_target_coordinates.begin() + m_space_dim * k + m_space_dim, m_source_coordinates.begin() + m_space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }

    // Matrix vector product, not necessary
    std::vector<double> operator*(std::vector<double> a) {
        std::vector<double> result(m_nr, 0);
        for (int i = 0; i < m_nr; i++) {
            for (int k = 0; k < m_nc; k++) {
                result[i] += this->get_coef(i, k) * a[k];
            }
        }
        return result;
    }

    // Frobenius norm, not necessary
    double normFrob() {
        double norm = 0;
        for (int i = 0; i < m_nr; i++) {
            for (int k = 0; k < m_nc; k++) {
                norm = norm + std::pow(this->get_coef(i, k), 2);
            }
        }
        return sqrt(norm);
    }
};

int main(int argc, char *argv[]) {

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Check the number of parameters
    if (argc < 3) {
        // Tell the user how to run the program
        cerr << "Usage: " << argv[0] << " distance \b outputfile \b outputpath" << endl;
        /* "Usage messages" are a conventional way of telling the user
         * how to run a program if they enter the command incorrectly.
         */
        return 1;
    }

    double distance        = StrToNbr<double>(argv[1]);
    std::string outputfile = argv[2];
    std::string outputpath = argv[3];

    // Parameters
    double epsilon  = 0.0001;
    int reqrank_max = 50;
    int nr          = 500;
    int nc          = 100;

    // Geometry
    int spatial_dimension = 3;
    vector<double> target_coordinates(spatial_dimension * nr);
    vector<double> source_coordinates(spatial_dimension * nc);
    create_sphere(nr, target_coordinates.data());
    create_sphere(nc, source_coordinates.data(), {distance, 0, 0});

    // Clustering
    ClusterTreeBuilder<double> recursive_build_strategy;
    Cluster<double> target_cluster = recursive_build_strategy.create_cluster_tree(nr, spatial_dimension, target_coordinates.data(), 2, 1);
    Cluster<double> source_cluster = recursive_build_strategy.create_cluster_tree(nc, spatial_dimension, source_coordinates.data(), 2, 1);

    MyMatrix A(spatial_dimension, target_cluster.get_permutation(), source_cluster.get_permutation(), target_coordinates, source_coordinates);
    double norm_A = A.normFrob();

    // SVD with fixed rank
    SVD<double> compressor_SVD;
    LowRankMatrix<double> A_SVD(A, compressor_SVD, target_cluster, source_cluster, reqrank_max, epsilon);
    std::vector<double> SVD_fixed_errors;
    for (int k = 0; k < A_SVD.rank_of() + 1; k++) {
        SVD_fixed_errors.push_back(Frobenius_absolute_error(target_cluster, source_cluster, A_SVD, A, k) / norm_A);
    }

    // fullACA with fixed rank
    fullACA<double> compressor_fullACA;
    LowRankMatrix<double> A_fullACA_fixed(A, compressor_fullACA, target_cluster, source_cluster, reqrank_max, epsilon);
    std::vector<double> fullACA_fixed_errors;
    for (int k = 0; k < A_fullACA_fixed.rank_of() + 1; k++) {
        fullACA_fixed_errors.push_back(Frobenius_absolute_error(target_cluster, source_cluster, A_fullACA_fixed, A, k) / norm_A);
    }

    // partialACA with fixed rank
    partialACA<double> compressor_partialACA;
    LowRankMatrix<double> A_partialACA_fixed(A, compressor_partialACA, target_cluster, source_cluster, reqrank_max, epsilon);
    std::vector<double> partialACA_fixed_errors;
    for (int k = 0; k < A_partialACA_fixed.rank_of() + 1; k++) {
        partialACA_fixed_errors.push_back(Frobenius_absolute_error(target_cluster, source_cluster, A_partialACA_fixed, A, k) / norm_A);
    }

    // sympartialACA with fixed rank
    sympartialACA<double> compressor_sympartialACA;
    LowRankMatrix<double> A_sympartialACA_fixed(A, compressor_sympartialACA, target_cluster, source_cluster, reqrank_max, epsilon);
    std::vector<double> sympartialACA_fixed_errors;
    for (int k = 0; k < A_sympartialACA_fixed.rank_of() + 1; k++) {
        sympartialACA_fixed_errors.push_back(Frobenius_absolute_error(target_cluster, source_cluster, A_sympartialACA_fixed, A, k) / norm_A);
    }

    // Output
    ofstream file_fixed((outputpath + "/" + outputfile).c_str());
    file_fixed << "Rank,SVD,Full ACA,partial ACA,sym partial ACA" << endl;
    for (int i = 0; i < reqrank_max; i++) {
        file_fixed << i << "," << SVD_fixed_errors[i] << "," << fullACA_fixed_errors[i] << "," << partialACA_fixed_errors[i] << "," << sympartialACA_fixed_errors[i] << endl;
    }

    // Finalize the MPI environment.
    MPI_Finalize();
}
