#include <htool/cluster_tree.hpp>
#include <htool/geometry.hpp>
#include <htool/output.hpp>



using namespace std;
using namespace htool;



int main(int argc, char* argv[]){
	MPI_Init(&argc,&argv);

	// Check the number of parameters
	if (argc < 3) {
		// Tell the user how to run the program
		cerr << "Usage: " << argv[0] << " depth \b inputname \b outputname" << endl;
		/* "Usage messages" are a conventional way of telling the user
		 * how to run a program if they enter the command incorrectly.
		 */
		return 1;
	}
	int depth = StrToNbr<int>(argv[1]);
	std::string inputname  = argv[2];
	std::string outputname = argv[3];

	std::vector<R3> x;

	Load_GMSH_nodes(x,inputname);

	Cluster_tree t(x);

	Write_gmsh_nodes(t.get_labels(depth),inputname,outputname);

	MPI_Finalize();
	return 0;
}
