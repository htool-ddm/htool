#ifndef USER_HPP
#define USER_HPP

// #include <iostream>
// #include <fstream>
#include <ctime>
#include <stack>
#include <string>
// #include <sstream>
// #include <vector>
// #include <cstdlib> // for exit()

using namespace std;

////////////////========================================================////////////////
////////////////////////========  Gestion temps	========////////////////////////////////

stack<clock_t> tictoc_stack;

void tic() {
	tictoc_stack.push(clock());
}

void toc() {
    double time =((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;
    cout << "Time elapsed: " << time << endl;
    tictoc_stack.pop();
}

void toc(vector<double>& times) {
	double time =((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;
	cout << "Time elapsed: " << time << endl;
	times.push_back(time);
	tictoc_stack.pop();
}

////////////////========================================================////////////////
////////////////////////========  Conversions	========////////////////////////////////

template <typename T>
string NbrToStr ( T Number )
{
	ostringstream ss;
	ss << Number;
	return ss.str();
}

template <typename T>
T StrToNbr ( const string &Text )
{
	istringstream ss(Text);
	T result;
	return ss >> result ? result : 0;
}


////////////////========================================================////////////////
////////////////////////========    String splitting	========////////////////////////

vector<string> &split(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}
vector<string> split(const string &s, char delim) {
	vector<string> elems;
	split(s, delim, elems);
	return elems;
}


#endif
