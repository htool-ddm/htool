#ifndef HTOOL_WRAPPER_MPI_HPP
#define HTOOL_WRAPPER_MPI_HPP

#include "../misc/define.hpp"
#include <limits.h>
#include <mpi.h>
#include <stdint.h>

namespace htool {
template <typename T>
struct wrapper_mpi {
    static MPI_Datatype mpi_type();
    static MPI_Datatype mpi_underlying_type() {
        return wrapper_mpi<underlying_type<T>>::mpi_type();
    }
};

template <>
inline MPI_Datatype wrapper_mpi<int>::mpi_type() { return MPI_INT; }
template <>
inline MPI_Datatype wrapper_mpi<float>::mpi_type() { return MPI_FLOAT; }
template <>
inline MPI_Datatype wrapper_mpi<double>::mpi_type() { return MPI_DOUBLE; }
template <>
inline MPI_Datatype wrapper_mpi<std::complex<float>>::mpi_type() { return MPI_C_COMPLEX; }
template <>
inline MPI_Datatype wrapper_mpi<std::complex<double>>::mpi_type() { return MPI_C_DOUBLE_COMPLEX; }

// https: //stackoverflow.com/questions/40807833/sending-size-t-type-data-with-mpi
#if SIZE_MAX == UCHAR_MAX
#    define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#    define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
#    define my_MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
#    define my_MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#    define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
#    error "what is happening here?"
#endif

} // namespace htool
#endif
