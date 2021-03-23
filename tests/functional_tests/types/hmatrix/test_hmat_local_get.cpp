#include <htool/clustering/ncluster.hpp>
#include <htool/htool.hpp>
#include <htool/lrmat/fullACA.hpp>
#include <htool/types/hmatrix.hpp>
using namespace std;
using namespace htool;

class MyMatrix : public IMatrix<double> {
    const vector<R3> &p1;
    const vector<R3> &p2;

  public:
    MyMatrix(const vector<R3> &p10, const vector<R3> &p20) : IMatrix(p10.size(), p20.size()), p1(p10), p2(p20) {}

    double get_coef(const int &i, const int &j) const { return 1. / (1e-5 + 4 * M_PI * norm2(p1[i] - p2[j])); }

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
    bool test           = 0;
    const int ndistance = 4;
    double distance[ndistance];
    distance[0] = 3;
    distance[1] = 5;
    distance[2] = 7;
    distance[3] = 10;
    SetNdofPerElt(1);
    SetEpsilon(1e-6);
    SetEta(0.1);

    for (int idist = 0; idist < ndistance; idist++) {

        srand(1);
        // we set a constant seed for rand because we want always the same result if we run the check many times
        // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

        int nr = 25;
        vector<int> Ir(nr); // row indices for the lrmatrix

        double z1 = 1;
        vector<R3> p1(nr);
        vector<double> r1(nr, 0);
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

        MyMatrix A(p1, p1);

        int size_numbering = p1.size() / size;
        int count_size     = 0;
        std::vector<std::pair<int, int>> MasterOffset;
        for (int p = 0; p < size - 1; p++) {
            MasterOffset.push_back(std::pair<int, int>(count_size, size_numbering));

            count_size += size_numbering;
        }
        MasterOffset.push_back(std::pair<int, int>(count_size, p1.size() - count_size));

        // local clustering
        std::shared_ptr<GeometricClustering> t = make_shared<GeometricClustering>();
        t->build_local_auto(p1, MasterOffset, 2);

        HMatrix<double, fullACA, GeometricClustering, RjasanowSteinbach> HA(A, t, p1);
        HA.print_infos();

        // Local diagonal
        std::vector<double> local_diagonal = HA.get_local_diagonal(true);
        double error                       = 0;
        for (int i = 0; i < MasterOffset[rank].second; i++) {
            error += std::pow(A.get_coef(i + MasterOffset[rank].first, i + MasterOffset[rank].first) - local_diagonal[i], 2);
            ;
        }

        if (rank == 0) {
            cout << "Error on local diagonal: " << std::sqrt(error) << endl;
        }
        test = test || !(std::sqrt(error) < 1e-10);

        // Local block
        Matrix<double> local_diagonal_block = HA.get_local_diagonal_block(true);
        error                               = 0;
        for (int j = 0; j < MasterOffset[rank].second; j++) {
            for (int i = 0; i < MasterOffset[rank].second; i++) {

                error += std::pow(A.get_coef(i + MasterOffset[rank].first, j + MasterOffset[rank].first) - local_diagonal_block(i, j), 2);
            }
        }

        if (rank == 0) {
            cout << "Error on local diagonal block: " << std::sqrt(error) << endl;
        }
        test = test || !(std::sqrt(error) < 1e-10);
    }
    if (rank == 0) {
        cout << "test: " << test << endl;
    }
    // Finalize the MPI environment.
    MPI_Finalize();

    return test;
}
