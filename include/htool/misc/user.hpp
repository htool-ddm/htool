#ifndef HTOOL_USER_HPP
#define HTOOL_USER_HPP

#include <cstddef> // for size_t
#include <numeric> // for accumulate
#include <sstream> // for basic_ostringstream, basic_istringstream, istri...
#include <string>  // for basic_string, allocator, string, char_traits
#include <vector>  // for vector

namespace htool {

// Conversions
template <typename T>
std::string NbrToStr(T Number) {
    std::ostringstream ss;
    ss << Number;
    return ss.str();
}

template <typename T>
T StrToNbr(const std::string &Text) {
    std::istringstream ss(Text);
    T result;
    return (ss >> result) ? result : 0;
}

//  String operations

inline std::vector<std::string> split(const std::string &s, const std::string &delim) {
    std::vector<std::string> elems;
    std::string line = s;
    std::size_t pos  = 0;
    std::string token;
    while ((pos = line.find(delim)) != std::string::npos) {
        token = line.substr(0, pos);
        elems.push_back(token);
        line.erase(0, pos + delim.length());
    }
    elems.push_back(line);
    return elems;
}

inline std::string join(std::string delimiter, std::vector<std::string> x) {
    return std::accumulate(x.begin(), x.end(), std::string(), [&](std::string ss, const std::string &s) { return ss.empty() ? s : ss + delimiter + s; });
}
} // namespace htool

#endif
