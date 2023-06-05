#include <complex>
#include <iostream>
#include <vector>

#include <htool/htool.hpp>

using namespace std;
using namespace htool;

class MyMatrix : public VirtualGenerator<double> {
    const vector<double> &p1;
    const vector<double> &p2;
    int space_dim;

  public:
    // Constructor
    MyMatrix(int space_dim0, int nr, int nc, const vector<double> &p10, const vector<double> &p20) : VirtualGenerator(nr, nc), p1(p10), p2(p20), space_dim(space_dim0) {}
    double get_coef(const int &k, const int &j) const {
        return 1. / (1 + std::sqrt(std::inner_product(p1.begin() + space_dim * k, p1.begin() + space_dim * k + space_dim, p2.begin() + space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }

    void copy_submatrix(int M, int N, const int *const rows, const int *const cols, double *ptr) const override {
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < N; k++) {
                ptr[j + M * k] = this->get_coef(rows[j], cols[k]);
            }
        }
    }

    std::vector<double> operator*(std::vector<double> a) {
        std::vector<double> result(nr, 0);
        for (int i = 0; i < nr; i++) {
            for (int k = 0; k < nc; k++) {
                result[i] += this->get_coef(i, k) * a[k];
            }
        }
        return result;
    }

    double normFrob() {
        double norm = 0;
        for (int i = 0; i < nr; i++) {
            for (int k = 0; k < nc; k++) {
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

    double epsilon  = 0.0001;
    int reqrank_max = 50;
    srand(1);
    // we set a constant seed for rand because we want always the same result if we run the check many times
    // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

    int nr = 500;
    int nc = 100;

    double z1 = 1;
    vector<double> p1(3 * nr);
    std::vector<int> tab1(nr);

    for (int j = 0; j < nr; j++) {

        double rho    = ((double)rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
        double theta  = ((double)rand() / (double)(RAND_MAX));
        p1[3 * j + 0] = sqrt(rho) * cos(2 * M_PI * theta);
        p1[3 * j + 1] = sqrt(rho) * sin(2 * M_PI * theta);
        p1[3 * j + 2] = z1;
        tab1[j]       = j;
    }
    // p2: points in a unit disk of the plane z=z2
    double z2 = 1 + distance;
    vector<double> p2(3 * nc);
    std::vector<int> tab2(nc);
    for (int j = 0; j < nc; j++) {
        double rho    = ((double)rand() / (RAND_MAX)); // (double) otherwise integer division!
        double theta  = ((double)rand() / (RAND_MAX));
        p2[3 * j + 0] = sqrt(rho) * cos(2 * M_PI * theta);
        p2[3 * j + 1] = sqrt(rho) * sin(2 * M_PI * theta);
        p2[3 * j + 2] = z2;
        tab2[j]       = j;
    }

    // Clustering

    Cluster<PCAGeometricClustering> t, s;
    t.build(nr, p1.data());
    s.build(nc, p2.data());

    std::shared_ptr<VirtualAdmissibilityCondition> AdmissibilityCondition = std::make_shared<RjasanowSteinbach>();
    Block<double> block(AdmissibilityCondition.get(), t, s);

    MyMatrix A(3, nr, nc, p1, p2);
    double norm_A = A.normFrob();

    // SVD with fixed rank
    SVD<double> compressor_SVD;
    LowRankMatrix<double> A_SVD(block, A, compressor_SVD, p1.data(), p2.data(), reqrank_max, epsilon);
    std::vector<double> SVD_fixed_errors;
    for (int k = 0; k < A_SVD.rank_of() + 1; k++) {
        SVD_fixed_errors.push_back(Frobenius_absolute_error(block, A_SVD, A, k) / norm_A);
    }
    std::cout << SVD_fixed_errors << std::endl;

    // fullACA with fixed rank
    fullACA<double> compressor_fullACA;
    LowRankMatrix<double> A_fullACA_fixed(block, A, compressor_fullACA, p1.data(), p2.data(), reqrank_max, epsilon);
    std::vector<double> fullACA_fixed_errors;
    for (int k = 0; k < A_fullACA_fixed.rank_of() + 1; k++) {
        fullACA_fixed_errors.push_back(Frobenius_absolute_error(block, A_fullACA_fixed, A, k) / norm_A);
    }
    std::cout << fullACA_fixed_errors << std::endl;

    // partialACA with fixed rank
    partialACA<double> compressor_partialACA;
    LowRankMatrix<double> A_partialACA_fixed(block, A, compressor_partialACA, p1.data(), p2.data(), reqrank_max, epsilon);
    std::vector<double> partialACA_fixed_errors;
    std::cout << A_partialACA_fixed.rank_of() << " " << reqrank_max << std::endl;
    for (int k = 0; k < A_partialACA_fixed.rank_of() + 1; k++) {
        partialACA_fixed_errors.push_back(Frobenius_absolute_error(block, A_partialACA_fixed, A, k) / norm_A);
    }

    std::cout << partialACA_fixed_errors << std::endl;
    // sympartialACA with fixed rank
    sympartialACA<double> compressor_sympartialACA;
    LowRankMatrix<double> A_sympartialACA_fixed(block, A, compressor_sympartialACA, p1.data(), p2.data(), reqrank_max, epsilon);
    std::vector<double> sympartialACA_fixed_errors;
    for (int k = 0; k < A_sympartialACA_fixed.rank_of() + 1; k++) {
        sympartialACA_fixed_errors.push_back(Frobenius_absolute_error(block, A_sympartialACA_fixed, A, k) / norm_A);
    }

    // Output
    ofstream file_fixed((outputpath + "/" + outputfile).c_str());
    file_fixed << "Rank,SVD,Full ACA,partial ACA,sym partial ACA" << endl;
    for (int i = 0; i < reqrank_max; i++) {
        file_fixed << i << "," << SVD_fixed_errors[i] << "," << fullACA_fixed_errors[i] << "," << partialACA_fixed_errors[i] << "," << sympartialACA_fixed_errors[i] << endl;
    }

    ofstream geometry_1((outputpath + "/geometry_1_" + outputfile).c_str());
    for (int i = 0; i < nr; i++) {
        geometry_1 << p1[i + 0] << "," << p1[i + 1] << "," << p1[i + 2] << endl;
    }

    ofstream geometry_2((outputpath + "/geometry_2_" + outputfile).c_str());
    for (int i = 0; i < nc; i++) {
        geometry_2 << p2[i + 0] << "," << p2[i + 1] << "," << p2[i + 2] << endl;
    }

    // Finalize the MPI environment.
    MPI_Finalize();
}
