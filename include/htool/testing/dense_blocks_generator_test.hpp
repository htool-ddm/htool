#include "../hmatrix/interfaces/virtual_dense_blocks_generator.hpp"
#include "../hmatrix/interfaces/virtual_generator.hpp"
#include <vector>

namespace htool {
template <typename CoefficientPrecision>
class DenseBlocksGeneratorTest : public VirtualDenseBlocksGenerator<CoefficientPrecision> {
  private:
    const VirtualGenerator<CoefficientPrecision> &m_generator;

  public:
    DenseBlocksGeneratorTest(const VirtualGenerator<CoefficientPrecision> &generator) : m_generator(generator) {}
    void copy_dense_blocks(const std::vector<int> &M, const std::vector<int> &N, const std::vector<int> &rows, const std::vector<int> &cols, std::vector<CoefficientPrecision *> &ptr) const override {
        for (int i = 0; i < M.size(); i++) {
            m_generator.copy_submatrix(M[i], N[i], rows[i], cols[i], ptr[i]);
        }
    }
};
} // namespace htool
