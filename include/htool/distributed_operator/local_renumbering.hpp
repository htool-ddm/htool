#ifndef HTOOL_LOCAL_RENUMBERING_HPP
#define HTOOL_LOCAL_RENUMBERING_HPP

#include "../clustering/cluster_node.hpp"

namespace htool {
class LocalRenumbering {
    int m_offset;
    int m_size;
    int m_global_size;
    const int *m_permutation;
    bool m_is_stable;

  public:
    LocalRenumbering(int offset, int size, int global_size, const int *permutation) : m_offset(offset), m_size(size), m_global_size(global_size), m_permutation(permutation), m_is_stable(true) {}
    template <typename CoordinatePrecision>
    LocalRenumbering(const Cluster<CoordinatePrecision> &cluster) : m_offset(cluster.get_offset()), m_size(cluster.get_size()), m_global_size(cluster.get_permutation().size()), m_permutation(cluster.get_permutation().data()), m_is_stable(cluster.is_root() || (is_cluster_on_partition(cluster) && cluster.is_permutation_local())) {}

    LocalRenumbering(const LocalRenumbering &)            = default;
    LocalRenumbering &operator=(const LocalRenumbering &) = default;
    LocalRenumbering(LocalRenumbering &&)                 = default;
    LocalRenumbering &operator=(LocalRenumbering &&)      = default;
    virtual ~LocalRenumbering() {}

    int get_offset() const { return m_offset; }
    int get_size() const { return m_size; }
    int get_global_size() const { return m_global_size; }
    const int *get_permutation() const { return m_permutation; }
    bool is_stable() const { return m_is_stable; }
};

template <typename CoefficientPrecision>
void htool_to_user_numbering(const LocalRenumbering &local_renumbering, const CoefficientPrecision *in, CoefficientPrecision *out) {
    if (!local_renumbering.is_stable()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Renumbering is not stable."); // LCOV_EXCL_LINE
    }
    const auto &permutation = local_renumbering.get_permutation();
    for (int i = 0; i < local_renumbering.get_size(); i++) {
        out[permutation[local_renumbering.get_offset() + i] - local_renumbering.get_offset()] = in[i];
    }
}

template <typename CoefficientPrecision>
void user_to_htool_numbering(const LocalRenumbering &local_renumbering, const CoefficientPrecision *in, CoefficientPrecision *out) {
    if (!local_renumbering.is_stable()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Renumbering is not stable."); // LCOV_EXCL_LINE
    }

    const auto &permutation = local_renumbering.get_permutation();
    for (int i = 0; i < local_renumbering.get_size(); i++) {
        out[i] = in[permutation[local_renumbering.get_offset() + i] - local_renumbering.get_offset()];
    }
}

} // namespace htool

#endif
