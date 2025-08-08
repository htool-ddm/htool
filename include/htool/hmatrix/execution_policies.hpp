#ifndef HTOOL_HMATRIX_EXECUTION_POLICIES_HPP
#define HTOOL_HMATRIX_EXECUTION_POLICIES_HPP

#include "hmatrix.hpp"
#include <vector>

#if defined(__cpp_lib_execution) && __cplusplus >= 201703L
#    include <execution>
namespace exec_compat {
using parallel_policy     = std::execution::parallel_policy;
inline constexpr auto par = std::execution::par;
using sequenced_policy    = std::execution::sequenced_policy;
inline constexpr auto seq = std::execution::seq;
} // namespace exec_compat
#else
namespace exec_compat {
struct sequenced_policy {};
static constexpr sequenced_policy seq{};
} // namespace exec_compat

namespace exec_compat {
struct parallel_policy {};
static constexpr parallel_policy par{};

} // namespace exec_compat
#endif

namespace htool {

// Base template: false by default
template <typename T>
struct is_execution_policy : std::false_type {};

// Specializations for fallback exec_compat policies
template <>
struct is_execution_policy<exec_compat::sequenced_policy> : std::true_type {};

template <>
struct is_execution_policy<exec_compat::parallel_policy> : std::true_type {};

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
struct omp_task_policy {
    // Shared state between tasks
    std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> L0;
    int max_nb_nodes = 64;
};

template <typename CoefficientPrecision, typename CoordinatePrecision>
struct is_execution_policy<omp_task_policy<CoefficientPrecision, CoordinatePrecision>> : std::true_type {};

template <typename T>
constexpr bool is_execution_policy_v = is_execution_policy<T>::value;

inline bool need_to_create_parallel_region() {
#if defined(_OPENMP)
    return omp_in_parallel() == 1 ? false : true;
#else
    return false;
#endif
}

} // namespace htool
#endif
