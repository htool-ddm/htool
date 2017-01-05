#ifndef USER_HPP
#define USER_HPP

// #include <iostream>
// #include <fstream>
#include <ctime>
#include <stack>
#include <string>
// #include <sstream>
// #include <std::vector>
// #include <cstdlib> // for exit()

namespace htool {
////////////////========================================================////////////////
////////////////////////========  Gestion temps	========////////////////////////////////

std::stack<clock_t> tictoc_stack;

void tic() {
	tictoc_stack.push(clock());
}

void toc() {
    double time =((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;
    std::cout << "Time elapsed: " << time << std::endl;
    tictoc_stack.pop();
}

void toc(std::vector<double>& times) {
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
