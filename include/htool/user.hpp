#ifndef HTOOL_USER_HPP
#define HTOOL_USER_HPP

#include <ctime>
#include <stack>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <mpi.h>

namespace htool {
////////////////========================================================////////////////
////////////////////////========  Gestion temps	========////////////////////////////////

std::stack<clock_t> tictoc_stack;

inline void tic(MPI_Comm comm= MPI_COMM_WORLD) {
	MPI_Barrier(comm);
	int rank;
	MPI_Comm_rank(comm, &rank);
	if (rank){
		tictoc_stack.push(clock());
	}
}

inline void toc(MPI_Comm comm = MPI_COMM_WORLD) {
		MPI_Barrier(comm);

		int rank;
		MPI_Comm_rank(comm, &rank);
		if (rank){
	    double time =((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;
	    std::cout << "Time elapsed: " << time << std::endl;
	    tictoc_stack.pop();
		}
}

inline void toc(std::vector<double>& times) {
	double time =((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;
	std::cout << "Time elapsed: " << time << std::endl;
	times.push_back(time);
	tictoc_stack.pop();
}

////////////////========================================================////////////////
////////////////////////========  Conversions	========////////////////////////////////

template <typename T>
std::string NbrToStr ( T Number )
{
	std::ostringstream ss;
	ss << Number;
	return ss.str();
}

template <typename T>
T StrToNbr ( const std::string &Text )
{
	std::istringstream ss(Text);
	T result;
	return ss >> result ? result : 0;
}


////////////////========================================================////////////////
////////////////////////========    String splitting	========////////////////////////

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}
std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}
}

#endif
