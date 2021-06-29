#include <htool/htool.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>

using namespace std;
using namespace htool;

template <typename ClusterImpl, template <typename> class LowRankMatrix>
int hmat(int argc, char *argv[]) {

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Check the number of parameters
    if (argc < 3) {
        // Tell the user how to run the program
        cerr << "Usage: " << argv[0] << " distance \b outputfile \b outputpath \b epsilon \b eta \b minclustersize \b nr \b nc" << endl;
        /* "Usage messages" are a conventional way of telling the user
		 * how to run a program if they enter the command incorrectly.
		 */
        return 1;
    }

    double distance        = StrToNbr<double>(argv[1]);
    std::string outputfile = argv[2];
    std::string outputpath = argv[3];
    double epsilon         = StrToNbr<double>(argv[4]);
    double eta             = StrToNbr<double>(argv[5]);
    double minclustersize  = StrToNbr<double>(argv[6]);
    int nr                 = StrToNbr<int>(argv[7]);
    int nc                 = StrToNbr<int>(argv[8]);

    //

    // Create points randomly
    srand(1);
    // we set a constant seed for rand because we want always the same result if we run the check many times
    // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

    vector<int> Ir(nr); // row indices for the lrmatrix
    vector<int> Ic(nc); // column indices for the lrmatrix

    double z1 = 1;
    vector<double> p1(3 * nr);
    double z2 = 1 + distance;
    vector<double> p2(3 * nc);
    create_disk(3, z1, nr, p1.data());
    create_disk(3, z2, nc, p2.data());

    // Matrix
    GeneratorTestDouble A(3, nr, nc, p1, p2);

    // Clustering
    std::shared_ptr<ClusterImpl> t = make_shared<ClusterImpl>();
    std::shared_ptr<ClusterImpl> s = make_shared<ClusterImpl>();
    t->build(nr, p1.data(), 2);
    s->build(nc, p2.data(), 2);
    t->set_minclustersize(minclustersize);
    s->set_minclustersize(minclustersize);

    // Hmatrix
    HMatrix<double, LowRankMatrix, RjasanowSteinbach> HA(t, s, epsilon, eta);
    HA.build_auto(A, p1.data(), p2.data());

    double mytime, maxtime, meantime;
    double meanmax(0), meanmean(0);

    // Global vectors
    std::vector<double> x_global(nc, 1), f_global(nr);

    // Global products
    for (int i = 0; i < 10; i++) {
        MPI_Barrier(HA.get_comm());
        mytime   = MPI_Wtime();
        f_global = HA * x_global;
        mytime   = MPI_Wtime() - mytime;
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, HA.get_comm());
        MPI_Reduce(&mytime, &meantime, 1, MPI_DOUBLE, MPI_SUM, 0, HA.get_comm());
        meantime /= size;
        if (i > 4) {
            meanmean += meantime;
            meanmax += maxtime;
        }
    }
    meanmax /= 5;
    meanmean /= 5;

    std::ofstream output;
    if (rank == 0) {
        output.open((outputpath + "/" + outputfile).c_str());
        output << "# Hmatrix" << std::endl;
    }
    HA.add_info("Mean_global_mat_vec_prod", NbrToStr(meanmean));
    HA.add_info("Max_global_mat_vec_prod", NbrToStr(meanmax));
    HA.save_infos((outputpath + "/" + outputfile).c_str(), std::ios::app, ": ");

    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}
