#ifndef HTOOL_WRAPPER_HPDDM_HPP
#define HTOOL_WRAPPER_HPDDM_HPP
#define HPDDM_NUMBERING 'F'
#define HPDDM_DENSE 1
#define HPDDM_FETI 0
#define HPDDM_BDD 0
#define LAPACKSUB
#define DLAPACK
#define EIGENSOLVER 1

#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wsign-compare"
#    pragma clang diagnostic ignored "-Wold-style-cast"
#    pragma clang diagnostic ignored "-Wshadow"
#    pragma clang diagnostic ignored "-Wdouble-promotion"
#    pragma clang diagnostic ignored "-Wunused-parameter"
#    pragma clang diagnostic ignored "-Wnon-virtual-dtor"
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wsign-compare"
#    pragma GCC diagnostic ignored "-Wold-style-cast"
#    pragma GCC diagnostic ignored "-Wshadow"
#    pragma GCC diagnostic ignored "-Wdouble-promotion"
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#    pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#    pragma GCC diagnostic ignored "-Wuseless-cast"
#    pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#    pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include <HPDDM.hpp>

#if defined(__clang__)
#    pragma clang diagnostic pop
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic pop
#endif

#include "../distributed_operator/distributed_operator.hpp" // for DistributedOperator
#include "../misc/misc.hpp"                                 // for conj_if_co...
#include "../solvers/interfaces/virtual_local_solver.hpp"
#include <algorithm> // for fill
#include <memory>    // for unique_ptr
#include <mpi.h>     // for MPI_Comm_rank
#include <vector>    // for vector

namespace htool {

template <typename CoefficientPrecision, template <class> class LocalSolver>
class DDM;

template <typename CoefficientPrecision>
class HPDDMCustomLocalSolver {
  private:
    std::unique_ptr<VirtualLocalSolver<CoefficientPrecision>> m_local_solver;

  public:
    HPDDMCustomLocalSolver() {}
    HPDDMCustomLocalSolver(const HPDDMCustomLocalSolver &) = delete;
    void set_local_solver(std::unique_ptr<VirtualLocalSolver<CoefficientPrecision>> local_solver) {
        m_local_solver = std::move(local_solver);
    }
    virtual ~HPDDMCustomLocalSolver() { dtor(); }
    static constexpr char numbering_ = 'C';
    void dtor() { m_local_solver.reset(nullptr); }
    template <char N = HPDDM_NUMBERING>
    void numfact(HPDDM::MatrixCSR<CoefficientPrecision> *const &dummy, bool boolean = false, CoefficientPrecision *const &ptr = nullptr) {
        m_local_solver->numfact(dummy, boolean, ptr);
    }
    void solve(CoefficientPrecision *const in, const unsigned short &a = 1) const {
        m_local_solver->solve(in, a);
    }
    void solve(const CoefficientPrecision *const in, CoefficientPrecision *const in2, const unsigned short &out = 1) const { m_local_solver->solve(in, in2, out); }
};

template <typename CoefficientPrecision, template <class> class LocalSolver>
class HPDDMOperator final : public HPDDM::Dense<LocalSolver, HPDDM::LapackTR, 'G', CoefficientPrecision> {
  protected:
    const DistributedOperator<CoefficientPrecision> *HA;
    std::vector<CoefficientPrecision> *in_global, *buffer;

  public:
    typedef HPDDM::Dense<LocalSolver, HPDDM::LapackTR, 'G', CoefficientPrecision> super;

    HPDDMOperator(const DistributedOperator<CoefficientPrecision> *A) : HA(A) {
        in_global = new std::vector<CoefficientPrecision>;
        buffer    = new std::vector<CoefficientPrecision>;
    }
    ~HPDDMOperator() {
        delete in_global;
        in_global = nullptr;
        delete buffer;
        buffer = nullptr;
    }

    int GMV(const CoefficientPrecision *const in, CoefficientPrecision *const out, const int &mu = 1) const override {
        int rankWorld;
        MPI_Comm_rank(HA->get_comm(), &rankWorld);
        int local_size = HA->get_target_partition().get_size_of_partition(rankWorld);

        // Tranpose without overlap
        if (mu != 1) {
            for (int i = 0; i < mu; i++) {
                for (int j = 0; j < local_size; j++) {
                    (*buffer)[i + j * mu] = in[i * this->getDof() + j];
                }
            }
            if (HA->get_symmetry_type() == 'H') {
                conj_if_complex(buffer->data(), local_size * mu);
            }
        }

        // All gather
        if (mu == 1) {
            HA->internal_vector_product_local_to_local(in, out, in_global->data());
        } else {
            HA->internal_matrix_product_local_to_local(buffer->data(), buffer->data() + local_size * mu, mu, in_global->data());
        }

        // Tranpose
        if (mu != 1) {
            if (HA->get_symmetry_type() == 'H') {
                conj_if_complex(buffer->data() + local_size * mu, local_size * mu);
            }
            for (int i = 0; i < mu; i++) {
                for (int j = 0; j < local_size; j++) {
                    out[i * this->getDof() + j] = (*buffer)[i + j * mu + local_size * mu];
                }
                std::fill(out + local_size + i * this->getDof(), out + (i + 1) * this->getDof(), 0);
            }
        } else {
            std::fill(out + local_size, out + this->getDof(), 0);
        }
        bool allocate = this->getMap().size() > 0 && this->getBuffer()[0] == nullptr ? this->setBuffer() : false;
        this->exchange(out, mu);
        if (allocate)
            this->clearBuffer(allocate);

        return 0;
    }

    void scaledexchange(CoefficientPrecision *const out, const int &mu = 1) const {
        this->template exchange<true>(out, mu);
    }

    void setType(typename super::Prcndtnr type) { this->type_ = type; }

    friend class DDM<CoefficientPrecision, LocalSolver>;
};

} // namespace htool

template <typename CoefficientPrecision, template <class> class LocalSolver>
struct HPDDM::hpddm_method_id<htool::HPDDMOperator<CoefficientPrecision, LocalSolver>> {
    static constexpr char value = HPDDM::hpddm_method_id<HPDDM::Dense<LocalSolver, HPDDM::LapackTR, 'G', CoefficientPrecision>>::value;
};
#endif
