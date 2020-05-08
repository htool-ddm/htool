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
//  Timing
std::stack<clock_t> htool_tictoc_stack;

inline void tic(MPI_Comm comm= MPI_COMM_WORLD) {
	MPI_Barrier(comm);
	int rank;
	MPI_Comm_rank(comm, &rank);
	if (rank==0){
		htool_tictoc_stack.push(clock());
	}
}

inline void toc(MPI_Comm comm = MPI_COMM_WORLD) {
		MPI_Barrier(comm);

		int rank;
		MPI_Comm_rank(comm, &rank);
		if (rank==0){
	    double time =((double)(clock() - htool_tictoc_stack.top())) / CLOCKS_PER_SEC;
	    std::cout << "Time elapsed: " << time << std::endl;
	    htool_tictoc_stack.pop();
		}
}

inline void toc(std::vector<double>& times) {
	double time =((double)(clock() - htool_tictoc_stack.top())) / CLOCKS_PER_SEC;
	std::cout << "Time elapsed: " << time << std::endl;
	times.push_back(time);
	htool_tictoc_stack.pop();
}

// Conversions
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


//  String operations

std::vector<std::string> split(const std::string &s, const std::string& delim) {
	std::vector<std::string> elems;
	std::string line = s ;
    size_t pos = 0;
    std::string token;
    while ((pos = line.find(delim)) != std::string::npos) {
        token = line.substr(0, pos);
        elems.push_back(token);
        line.erase(0, pos + delim.length());
    }
    elems.push_back(line);
	return elems;
}

std::string join(std::string delimiter, std::vector<std::string> x){
	return std::accumulate(std::begin(x), std::end(x), std::string(),[&](std::string &ss, std::string &s){return ss.empty() ? s : ss + delimiter + s;});
}
}

// Number of instances
// http://www.drdobbs.com/cpp/counting-objects-in-c/184403484?pgno=2
template<typename T>
class Counter {
public:
    Counter() { ++count; }
    Counter(const Counter&) { ++count; }
    ~Counter() { --count; }

    static size_t howMany()
    { return count; }

private:
    static size_t count;
};

template<typename T>
size_t
Counter<T>::count = 0;

#endif
