#ifndef HTOOL_HMATRIX_LINALG_EXECUTOR_THREAD_EXECUTOR_HPP
#define HTOOL_HMATRIX_LINALG_EXECUTOR_THREAD_EXECUTOR_HPP

#include "../../../basic_types/tree.hpp" // for preorder_tree_traversal
#include "../../../misc/logger.hpp"      // for Logger
#include "../../hmatrix.hpp"             // for HMatrix
#include <algorithm>                     // for max
#include <atomic>                        // for atomic
#include <condition_variable>            // for condition_variable
#include <deque>                         // for deque
#include <exception>                     // for exception_ptr
#include <functional>                    // for function
#include <mutex>                         // for mutex, lock_guard, unique_lock
#include <thread>                        // for thread
#include <unordered_map>                 // for unordered_map
#include <utility>                       // for move
#include <vector>                        // for vector

namespace htool {

/**
 * @brief An object accessed by a task spawned on an executor: either a node of an HMatrix, or another object identified by its address
 * (e.g. a low-rank term computed by one task and read by others).
 *
 * Writes must be to leaves of an HMatrix or to other objects. A read of an internal node of an HMatrix is a read of all its leaves.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision>
struct ExecutorAccess {
    const void *key;
    const HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix;

    ExecutorAccess(const HMatrix<CoefficientPrecision, CoordinatePrecision> &node) : key(&node), hmatrix(&node) {}
    explicit ExecutorAccess(const void *object) : key(object), hmatrix(nullptr) {}
};

/**
 * @brief Executor running every task immediately, in the calling thread, so that a driver written for an executor (see ThreadExecutor)
 * runs sequentially.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class SerialExecutor {
  public:
    using Access = ExecutorAccess<CoefficientPrecision, CoordinatePrecision>;

    template <typename Work>
    void spawn(Work &&work, const std::vector<Access> &, const std::vector<Access> &) { work(); }

    void wait() {}
};

/**
 * @brief Executor running tasks on a pool of threads, each task waiting only for the last task which wrote each of the objects it reads or
 * writes.
 *
 * This is the C++ counterpart of the ThreadsExecutor of HMatrices.jl. There is no general dependency analysis (no comparison of the accesses
 * of a new task with those of the tasks already spawned, as done by an OpenMP runtime with depend clauses): the executor only keeps the
 * last writer of every leaf and object, which is enough for drivers satisfying the following access pattern:
 *   - a leaf (or object) is written by a sequence of tasks, which are run in the order in which they are spawned,
 *   - a leaf (or object) is only read once final, i.e. after its last write has been spawned, so that there are no write-after-read hazards.
 * A read of an internal node waits for a join over the last writers of its leaves, created at its first read and reused afterwards.
 *
 * Tasks never block: each one counts its unfinished dependencies, and is put in the ready queue by the task completing the last of them. The
 * thread calling spawn is the driver: it only builds the graph, and works on the tasks once it calls wait, which returns when all the tasks
 * spawned are done, and rethrows the first exception thrown by one of them.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class ThreadExecutor {
  public:
    using Access      = ExecutorAccess<CoefficientPrecision, CoordinatePrecision>;
    using HMatrixType = HMatrix<CoefficientPrecision, CoordinatePrecision>;

    /**
     * @param number_of_threads Number of threads working on the tasks, including the driver once it calls wait.
     * @param check_accesses Whether to check that the driver satisfies the access pattern above (for testing: it is costly).
     */
    explicit ThreadExecutor(int number_of_threads, bool check_accesses = false) : m_check_accesses(check_accesses) {
        for (int i = 1; i < std::max(number_of_threads, 1); i++) {
            m_workers.emplace_back([this]() { work(false); });
        }
    }

    ThreadExecutor(const ThreadExecutor &)            = delete;
    ThreadExecutor &operator=(const ThreadExecutor &) = delete;

    ~ThreadExecutor() {
        {
            std::lock_guard<std::mutex> lock(m_queue_mutex);
            m_stop = true;
        }
        m_queue_condition.notify_all();
        for (auto &worker : m_workers) {
            worker.join();
        }
    }

    template <typename Work>
    void spawn(Work &&work, const std::vector<Access> &reads, const std::vector<Access> &writes) {
        Task *task = new_task(std::function<void()>(std::forward<Work>(work)));
        for (const Access &access : reads) {
            add_dependency(*task, last_writer(access));
        }
        for (const Access &access : writes) {
            if (access.hmatrix != nullptr && !access.hmatrix->is_leaf()) {
                Logger::get_instance().log(LogLevel::ERROR, "ThreadExecutor: a task writes an internal node of an HMatrix."); // LCOV_EXCL_LINE
            }
            if (m_check_accesses) {
                check_write(access);
            }
            add_dependency(*task, last_writer(access));
            m_last_writers[access.key] = task;
        }
        release(*task);
    }

    /// Works on the tasks until all the tasks spawned are done, then forgets them (and the last writers).
    void wait() {
        work(true);
        m_tasks.clear();
        m_last_writers.clear();
        m_joined_nodes.clear();
        if (m_exception) {
            std::exception_ptr exception = m_exception;
            m_exception                  = nullptr;
            std::rethrow_exception(exception);
        }
    }

  private:
    struct Task {
        std::function<void()> work;
        // One more than the number of unfinished dependencies while the task is being spawned, so that it is not run before.
        std::atomic<int> number_of_unfinished_dependencies{1};
        std::mutex mutex; // protects is_done and successors
        bool is_done = false;
        std::vector<Task *> successors;

        explicit Task(std::function<void()> &&task_work) : work(std::move(task_work)) {}
    };

    // Accessed only by the driver
    std::deque<Task> m_tasks; // stable addresses
    std::unordered_map<const void *, Task *> m_last_writers;
    std::vector<const HMatrixType *> m_joined_nodes; // internal nodes read, kept only if m_check_accesses
    bool m_check_accesses;

    // Shared with the workers
    std::mutex m_queue_mutex;
    std::condition_variable m_queue_condition;
    std::deque<Task *> m_ready_tasks;
    std::atomic<std::size_t> m_number_of_unfinished_tasks{0};
    bool m_stop = false;
    std::mutex m_exception_mutex;
    std::exception_ptr m_exception;
    std::vector<std::thread> m_workers;

    Task *new_task(std::function<void()> &&work) {
        m_tasks.emplace_back(std::move(work));
        m_number_of_unfinished_tasks++;
        return &m_tasks.back();
    }

    // The task after which the object accessed is up to date, if any. A read of an internal node, which is final, waits for a join over
    // the last writers of its leaves, created once.
    Task *last_writer(const Access &access) {
        auto it = m_last_writers.find(access.key);
        if (it != m_last_writers.end()) {
            return it->second;
        }
        if (access.hmatrix == nullptr || access.hmatrix->is_leaf()) {
            return nullptr;
        }
        std::vector<Task *> writers;
        preorder_tree_traversal(*access.hmatrix, [&](const HMatrixType &node) {
            if (node.is_leaf()) {
                auto leaf_it = m_last_writers.find(&node);
                if (leaf_it != m_last_writers.end()) {
                    writers.push_back(leaf_it->second);
                }
            }
        });
        Task *join = nullptr;
        if (!writers.empty()) {
            join = new_task(std::function<void()>());
            for (Task *writer : writers) {
                add_dependency(*join, writer);
            }
            release(*join);
        }
        m_last_writers[access.key] = join;
        if (m_check_accesses) {
            m_joined_nodes.push_back(access.hmatrix);
        }
        return join;
    }

    // A write to a leaf below an internal node already read would not be waited for by the readers of this node.
    void check_write(const Access &access) const {
        if (access.hmatrix == nullptr) {
            return;
        }
        for (const HMatrixType *joined_node : m_joined_nodes) {
            if (left_cluster_contains_right_cluster(joined_node->get_target_cluster(), access.hmatrix->get_target_cluster()) && left_cluster_contains_right_cluster(joined_node->get_source_cluster(), access.hmatrix->get_source_cluster())) {
                Logger::get_instance().log(LogLevel::ERROR, "ThreadExecutor: a leaf is written after an internal node containing it was read."); // LCOV_EXCL_LINE
            }
        }
    }

    void add_dependency(Task &task, Task *dependency) {
        if (dependency == nullptr || dependency == &task) {
            return;
        }
        std::lock_guard<std::mutex> lock(dependency->mutex);
        if (!dependency->is_done) {
            task.number_of_unfinished_dependencies++;
            dependency->successors.push_back(&task);
        }
    }

    // Decrements the number of unfinished dependencies of task, and makes it ready once it reaches zero.
    void release(Task &task) {
        if (--task.number_of_unfinished_dependencies == 0) {
            {
                std::lock_guard<std::mutex> lock(m_queue_mutex);
                m_ready_tasks.push_back(&task);
            }
            m_queue_condition.notify_one();
        }
    }

    void execute(Task &task) {
        if (task.work) {
            try {
                task.work();
            } catch (...) {
                std::lock_guard<std::mutex> lock(m_exception_mutex);
                if (!m_exception) {
                    m_exception = std::current_exception();
                }
            }
            // A finished task must not keep alive what its work captured (e.g. low-rank terms)
            task.work = nullptr;
        }
        std::vector<Task *> successors;
        {
            std::lock_guard<std::mutex> lock(task.mutex);
            task.is_done = true;
            successors.swap(task.successors);
        }
        for (Task *successor : successors) {
            release(*successor);
        }
        if (--m_number_of_unfinished_tasks == 0) {
            // Under the lock, so that the driver cannot miss the notification between its check and its wait
            std::lock_guard<std::mutex> lock(m_queue_mutex);
            m_queue_condition.notify_all();
        }
    }

    // Runs the ready tasks, until all the tasks spawned are done (the driver, in wait) or until the executor is destroyed (the workers).
    void work(bool is_driver) {
        while (true) {
            Task *task = nullptr;
            {
                std::unique_lock<std::mutex> lock(m_queue_mutex);
                m_queue_condition.wait(lock, [&]() { return !m_ready_tasks.empty() || (is_driver ? m_number_of_unfinished_tasks.load() == 0 : m_stop); });
                if (m_ready_tasks.empty()) {
                    return;
                }
                task = m_ready_tasks.front();
                m_ready_tasks.pop_front();
            }
            execute(*task);
        }
    }
};

} // namespace htool

#endif
