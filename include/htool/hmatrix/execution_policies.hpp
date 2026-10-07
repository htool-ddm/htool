#ifndef HTOOL_HMATRIX_EXECUTION_POLICIES_HPP
#define HTOOL_HMATRIX_EXECUTION_POLICIES_HPP

#include "hmatrix.hpp"
#include "htool/hmatrix/task_dependencies.hpp"
#include <vector>
#if defined(HTOOL_WITH_STD_EXECUTION_API) && HTOOL_WITH_STD_EXECUTION_API && __has_include(<execution>)
#    include <execution>
#endif
#if defined(HTOOL_WITH_STD_EXECUTION_API) && HTOOL_WITH_STD_EXECUTION_API && __has_include(<execution>) && defined(__cpp_lib_execution) && __cplusplus >= 201703L
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

/**
 * @brief Execution policy for OpenMP task-based build and factorizations.
 *
 * Stores the L0 used as task dependencies. build always sets L0 for the new
 * HMatrix. The factorizations keep L0 if it is a cut of the block tree of the
 * factorized HMatrix (for example the L0 set by its build, or one assigned
 * directly), and recompute it otherwise; call set_L0 explicitly after changing
 * max_number_of_nodes or cost_function.
 *
 * If called outside a parallel region, build, lu_factorization,
 * cholesky_factorization and ldlt_factorization create one and return once all
 * their tasks are done.
 * If L0 is reduced to the root, they run sequentially.
 *
 * If called inside a parallel region (e.g. in a single construct), they only
 * create tasks and return before these tasks are completed, so that tasks of
 * successive calls can overlap through their dependencies. Until a taskwait
 * (or the end of the enclosing region):
 *   - the HMatrix is only complete after a taskwait: before that, it can only be
 *     passed to other task-based calls with the same policy (hence the same L0),
 *     whose tasks wait for the ones they need through their dependencies,
 *   - the HMatrix can be moved, but must not be destroyed while its tasks are pending,
 *   - the generator passed to build must stay alive.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
struct omp_task_policy {
    // Shared state between tasks
    HMatrixTaskDependencies<CoefficientPrecision, CoordinatePrecision> hmatrix_task_dependencies;
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

// Call function, which creates tasks, in a parallel region with a single thread creating them. If already in a parallel region, function is called directly.
template <typename Function>
void run_in_task_region(Function &&function) {
    if (need_to_create_parallel_region()) {
#if defined(_OPENMP)
#    pragma omp parallel
#    pragma omp single
#endif
        function();
    } else {
        function();
    }
}

// Recompute the L0 of hmatrix_task_dependencies if it is not a cut of the block tree of hmatrix.
// In a parallel region, pending tasks may depend on the nodes of the previous L0, which the tasks using the new one would not wait for: wait for them first.
template <typename CoefficientPrecision, typename CoordinatePrecision>
void update_L0(HMatrixTaskDependencies<CoefficientPrecision, CoordinatePrecision> &hmatrix_task_dependencies, HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
    if (!hmatrix_task_dependencies.is_L0_of(hmatrix)) {
        if (!hmatrix_task_dependencies.L0.empty() && !need_to_create_parallel_region()) {
#if defined(_OPENMP)
#    pragma omp taskwait
#endif
        }
        hmatrix_task_dependencies.set_L0(hmatrix);
    }
}

// Call task_based_function(L0) in a task region, or sequential_function() if L0 is reduced to hmatrix: a single task brings no parallelism,
// and hmatrix could be moved while the task is pending.
template <typename CoefficientPrecision, typename CoordinatePrecision, typename SequentialFunction, typename TaskBasedFunction>
void run_task_based(HMatrixTaskDependencies<CoefficientPrecision, CoordinatePrecision> &hmatrix_task_dependencies, const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, SequentialFunction &&sequential_function, TaskBasedFunction &&task_based_function) {
    auto &L0 = hmatrix_task_dependencies.L0;
    if (L0.size() == 1 && L0[0] == &hmatrix) {
        sequential_function();
    } else {
        run_in_task_region([&]() { task_based_function(L0); });
    }
}

} // namespace htool
#endif
