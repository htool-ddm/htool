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
 *   - the generator passed to build must stay alive, and so must the
 *     HMatrixTreeBuilder if the generator is a VirtualGenerator, since the
 *     builder holds the internal generator wrapping it.
 *
 * lu_solve, cholesky_solve and ldlt_solve (and their internal_ versions) use the
 * L0 of the factorized HMatrix in the same way, but always return once their
 * tasks are done; in a parallel region, their tasks first wait for the pending
 * tasks of the factorization with the same policy. The task_based_internal_*_solve_row_major
 * functions, called with that L0, only create these tasks, like the factorizations:
 * their right-hand side is only complete after a taskwait.
 *
 * add_hmatrix_vector_product and add_hmatrix_matrix_product use the L0 in the
 * same way as the solves, with target_L0 and source_L0 as dependencies on the
 * rows of their output and input.
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

// True outside any parallel region. omp_in_parallel() is not used: it is false in a parallel region run by a single thread
// (e.g. with OMP_NUM_THREADS=1), where a new region would not wait for the pending tasks of the enclosing one.
inline bool need_to_create_parallel_region() {
#if defined(_OPENMP)
    return omp_get_level() == 0;
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

// Recompute the L0 of hmatrix_task_dependencies if it is not a cut of the block tree of hmatrix, and its cluster L0s if they are not cuts
// of the cluster trees of hmatrix (e.g. after L0 is assigned directly).
// In a parallel region, pending tasks may depend on the previous ones, which the tasks using the new ones would not wait for: wait for them first.
template <typename CoefficientPrecision, typename CoordinatePrecision>
void update_L0(HMatrixTaskDependencies<CoefficientPrecision, CoordinatePrecision> &hmatrix_task_dependencies, const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
    bool is_L0_of_hmatrix           = hmatrix_task_dependencies.is_L0_of(hmatrix);
    bool are_cluster_L0s_of_hmatrix = hmatrix_task_dependencies.are_cluster_L0s_of(hmatrix);
    if (is_L0_of_hmatrix && are_cluster_L0s_of_hmatrix) {
        return;
    }
    if (!hmatrix_task_dependencies.L0.empty() && !need_to_create_parallel_region()) {
#if defined(_OPENMP)
#    pragma omp taskwait
#endif
    }
    if (!is_L0_of_hmatrix) {
        hmatrix_task_dependencies.set_L0(hmatrix);
    } else {
        hmatrix_task_dependencies.set_cluster_L0s(hmatrix);
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
