#ifndef PRECONDITIONER_HPP
#define PRECONDITIONER_HPP

#include "matrix.hpp"
#include "lapack.hpp"
#include "wrapper_mpi.hpp"

namespace htool{


template<typename T>
class Preconditioner{
private:
  Preconditioner(const Preconditioner&) = default; // copy constructor
  Preconditioner& operator=(const Preconditioner&) = default; // copy assignement operator

protected:
  // Data member
  int n_local;
  MPI_Comm comm;

  Preconditioner() = delete;
  Preconditioner(int n0):n_local(n0){}


public:
  Preconditioner(Preconditioner&&) = default; // move constructor
  Preconditioner& operator=(Preconditioner&&) = default; // move assignement operator

  virtual void apply(const T* const in, T* const out);
};

template<typename T>
class Identity : public Preconditioner<T>{
public:
  Identity(int n0):Preconditioner<T>(n0){};
  void apply(const T* const in, T* const out){
    std::copy_n(in, this->n_local, out);
  }


};


}
#endif
