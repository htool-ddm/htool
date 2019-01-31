#ifndef HTOOL_WRAPPER_MPI_HPP
#define HTOOL_WRAPPER_MPI_HPP

#include <mpi.h>

namespace htool{
template<typename T>
struct wrapper_mpi{
  static MPI_Datatype mpi_type();
};

template<>
inline MPI_Datatype wrapper_mpi<int>::mpi_type() { return MPI_INT; }
template<>
inline MPI_Datatype wrapper_mpi<float>::mpi_type() { return MPI_FLOAT; }
template<>
inline MPI_Datatype wrapper_mpi<double>::mpi_type() { return MPI_DOUBLE; }
template<>
inline MPI_Datatype wrapper_mpi<std::complex<float>>::mpi_type() { return MPI_COMPLEX; }
template<>
inline MPI_Datatype wrapper_mpi<std::complex<double>>::mpi_type() { return MPI_DOUBLE_COMPLEX; }

}
#endif
