#ifndef HTOOL_WRAPPER_HPDDM_HPP
#define HTOOL_WRAPPER_HPDDM_HPP

#define HPDDM_NUMBERING 'F'
#define HPDDM_DENSE 1
#define HPDDM_FETI 0
#define HPDDM_BDD 0
#define LAPACKSUB
#define DLAPACK
#define EIGENSOLVER 1
// #include "../solvers/proto_ddm.hpp"
#include "../types/matrix.hpp"
#include "../types/virtual_hmatrix.hpp"
#include <HPDDM.hpp>

namespace htool {

template <typename T>
class DDM;

template <typename T>
class HPDDMDense : public HpDense<T, 'G'> {
  protected:
    const VirtualHMatrix<T> *const HA;
    std::vector<T> *in_global, *buffer;

  public:
    typedef HpDense<T, 'G'> super;

    HPDDMDense(const VirtualHMatrix<T> *const A) : HA(A) {
        in_global = new std::vector<T>;
        buffer    = new std::vector<T>;
    }
    ~HPDDMDense() {
        delete in_global;
        in_global = nullptr;
        delete buffer;
        buffer = nullptr;
    }

    virtual int GMV(const T *const in, T *const out, const int &mu = 1) const override {
        int local_size = HA->get_local_size();

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
        if (mu == 1) { // C'est moche
            HA->mymvprod_local_to_local(in, out, mu, in_global->data());
        } else {
            HA->mymvprod_local_to_local(buffer->data(), buffer->data() + local_size * mu, mu, in_global->data());
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

    void scaledexchange(T *const out, const int &mu = 1) const {
        this->template exchange<true>(out, mu);
    }

    void setType(typename super::Prcndtnr type) { this->_type = type; };

    friend class DDM<T>;
};

} // namespace htool

template <typename T>
struct HPDDM::hpddm_method_id<htool::HPDDMDense<T>> { static constexpr char value = HPDDM::hpddm_method_id<HpDense<T, 'G'>>::value; };
#endif
