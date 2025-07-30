#ifndef HTOOL_DISTRIBUTED_OPERATOR_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_HPP

#include "../misc/logger.hpp"                              // for Logger, LogLevel
#include "../misc/misc.hpp"                                // for conj_if_complex
#include "../misc/user.hpp"                                // for NbrToStr, StrToNbr
#include "../wrappers/wrapper_mpi.hpp"                     // for wrapper_mpi
#include "interfaces/virtual_global_to_local_operator.hpp" // for VirtualLocalToLocalOperator
#include "interfaces/virtual_local_to_local_operator.hpp"  // for VirtualGlobalToLocalOperator
#include "interfaces/virtual_partition.hpp"                // for IPartition
#include <algorithm>                                       // for fill, copy_n, fill_n, transform
#include <functional>                                      // for plus
#include <map>                                             // for map
#include <mpi.h>                                           // for MPI_Comm_rank, MPI_Comm_size
#include <string>                                          // for basic_string, operator<, string
#include <vector>                                          // for vector

namespace htool {
template <typename CoefficientPrecision>
class DistributedOperator {

  private:
    //
    const VirtualPartition<CoefficientPrecision> &m_target_partition;
    const VirtualPartition<CoefficientPrecision> &m_source_partition;

    // Local operators
    std::vector<const VirtualLocalToLocalOperator<CoefficientPrecision> *> m_local_to_local_operators   = {};
    std::vector<const VirtualGlobalToLocalOperator<CoefficientPrecision> *> m_global_to_local_operators = {};

    // Properties
    MPI_Comm m_comm = MPI_COMM_WORLD;

    //

  public:
    // no copy
    DistributedOperator(const DistributedOperator &)                       = delete;
    DistributedOperator &operator=(const DistributedOperator &)            = delete;
    DistributedOperator(DistributedOperator &&cluster) noexcept            = default;
    DistributedOperator &operator=(DistributedOperator &&cluster) noexcept = default;
    virtual ~DistributedOperator()                                         = default;

    // Constructor
    explicit DistributedOperator(const VirtualPartition<CoefficientPrecision> &target_partition, const VirtualPartition<CoefficientPrecision> &source_partition, MPI_Comm comm) : m_target_partition(target_partition), m_source_partition(source_partition), m_comm(comm) {}

    void add_local_to_local_operator(const VirtualLocalToLocalOperator<CoefficientPrecision> *local_operator) {
        m_local_to_local_operators.push_back(local_operator);
    }

    void add_global_to_local_operator(const VirtualGlobalToLocalOperator<CoefficientPrecision> *local_operator) {
        m_global_to_local_operators.push_back(local_operator);
    }

    // Getters
    MPI_Comm get_comm() const { return m_comm; }
    const VirtualPartition<CoefficientPrecision> &get_target_partition() const { return m_target_partition; }
    const VirtualPartition<CoefficientPrecision> &get_source_partition() const { return m_source_partition; }
    const std::vector<const VirtualLocalToLocalOperator<CoefficientPrecision> *> get_local_to_local_operators() const { return m_local_to_local_operators; }
    const std::vector<const VirtualGlobalToLocalOperator<CoefficientPrecision> *> get_global_to_local_operators() const { return m_global_to_local_operators; }
};

} // namespace htool

#endif
