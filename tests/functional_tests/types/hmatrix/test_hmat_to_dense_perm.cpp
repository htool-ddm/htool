#include <htool/clustering/ncluster.hpp>
#include <htool/lrmat/fullACA.hpp>
#include <htool/types/hmatrix.hpp>

using namespace std;
using namespace htool;

class MyMatrix : public IMatrix<double> {
    const vector<R3> &p1;
    const vector<R3> &p2;

  public:
    MyMatrix(const vector<R3> &p10, const vector<R3> &p20) : IMatrix(p10.size(), p20.size()), p1(p10), p2(p20) {}

    double get_coef(const int &i, const int &j) const { return 1. / (4 * M_PI * norm2(p1[i] - p2[j])); }

    std::vector<double> operator*(std::vector<double> a) {
        std::vector<double> result(p1.size(), 0);
        for (int i = 0; i < p1.size(); i++) {
            for (int k = 0; k < p2.size(); k++) {
                result[i] += this->get_coef(i, k) * a[k];
            }
        }
        return result;
    }
};

int main(int argc, char *argv[]) {

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //
    bool test       = 0;
    double distance = 1;
    SetNdofPerElt(1);
    SetEpsilon(1e-6);
    SetEta(0.1);
    SetMinClusterSize(2);

    srand(1);
    // we set a constant seed for rand because we want always the same result if we run the check many times
    // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

    int nr = 20;
    int nc = 10;
    vector<int> Ir(nr); // row indices for the lrmatrix
    vector<int> Ic(nc); // column indices for the lrmatrix

    double z1 = 1;
    vector<R3> p1(nr);
    vector<double> r1(nr, 0);
    vector<double> g1(nr, 1);
    vector<int> tab1(nr);
    for (int j = 0; j < nr; j++) {
        Ir[j]        = j;
        double rho   = ((double)rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
        double theta = ((double)rand() / (double)(RAND_MAX));
        p1[j][0]     = sqrt(rho) * cos(2 * M_PI * theta);
        p1[j][1]     = sqrt(rho) * sin(2 * M_PI * theta);
        p1[j][2]     = z1;
        // sqrt(rho) otherwise the points would be concentrated in the center of the disk
        tab1[j] = j;
    }
    // p2: points in a unit disk of the plane z=z2
    double z2 = 1 + distance;
    vector<R3> p2(nc);
    vector<double> r2(nc, 0);
    vector<double> g2(nc, 1);
    vector<int> tab2(nc);
    for (int j = 0; j < nc; j++) {
        Ic[j]        = j;
        double rho   = ((double)rand() / (RAND_MAX)); // (double) otherwise integer division!
        double theta = ((double)rand() / (RAND_MAX));
        p2[j][0]     = sqrt(rho) * cos(2 * M_PI * theta);
        p2[j][1]     = sqrt(rho) * sin(2 * M_PI * theta);
        p2[j][2]     = z2;
        tab2[j]      = j;
    }

    MyMatrix A(p1, p2);

    int size_numbering = p1.size() / size;
    int count_size     = 0;
    std::vector<std::pair<int, int>> MasterOffset_target;
    for (int p = 0; p < size - 1; p++) {
        MasterOffset_target.push_back(std::pair<int, int>(count_size, size_numbering));

        count_size += size_numbering;
    }
    MasterOffset_target.push_back(std::pair<int, int>(count_size, p1.size() - count_size));

    size_numbering = p2.size() / size;
    count_size     = 0;
    std::vector<std::pair<int, int>> MasterOffset_source;
    for (int p = 0; p < size - 1; p++) {
        MasterOffset_source.push_back(std::pair<int, int>(count_size, size_numbering));

        count_size += size_numbering;
    }
    MasterOffset_source.push_back(std::pair<int, int>(count_size, p2.size() - count_size));

    // Hmatrix
    std::shared_ptr<GeometricClustering> t = make_shared<GeometricClustering>();
    std::shared_ptr<GeometricClustering> s = make_shared<GeometricClustering>();
    t->build_local_auto(p1, MasterOffset_target);
    s->build_local_auto(p2, MasterOffset_source);
    HMatrix<double, fullACA, GeometricClustering, RjasanowSteinbach> HA(A, t, p1, tab1, s, p2, tab2);
    HA.print_infos();

    // Dense Matrix
    Matrix<double> DA       = HA.to_dense_perm();
    Matrix<double> DA_local = HA.to_local_dense_perm();

    // Test dense matrices
    double error = 0;
    for (int i = 0; i < t->get_local_size(); i++) {
        for (int j = 0; j < nc; j++) {
            error += std::abs(DA(i + t->get_local_offset(), j) - DA_local(i, j));
        }
    }

    if (rank == 0) {
        cout << "Difference between dense matrix and local dense matrix: " << error << endl;
    }
    test = test || !(error < 1e-10);

    // Finalize the MPI environment.
    MPI_Finalize();

    return test;
}
