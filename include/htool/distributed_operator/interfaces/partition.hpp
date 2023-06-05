#ifndef HTOOL_DISTIRBUTED_OPERATOR_PARTITION_HPP
#define HTOOL_DISTIRBUTED_OPERATOR_PARTITION_HPP

namespace htool {

template <typename CoefficientPrecision>
class IPartition {
  public:
    virtual int get_size_of_partition(int subdomain_number) const   = 0;
    virtual int get_offset_of_partition(int subdomain_number) const = 0;

    virtual int get_global_size() const = 0;

    virtual void global_to_partition_numbering(const CoefficientPrecision *const in, CoefficientPrecision *const out) const = 0;
    virtual void partition_to_global_numbering(const CoefficientPrecision *const in, CoefficientPrecision *const out) const = 0;

    virtual void local_to_local_partition_numbering(int subdomain_number, const CoefficientPrecision *const in, CoefficientPrecision *const out) const = 0;
    virtual void local_partition_to_local_numbering(int subdomain_number, const CoefficientPrecision *const in, CoefficientPrecision *const out) const = 0;

    virtual bool is_renumbering_local() const = 0;

    virtual ~IPartition() {}
};

} // namespace htool
#endif
