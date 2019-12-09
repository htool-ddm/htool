#
# Try to find ARPACK library and include path.
# Once done this will define
#
# ARPACK_FOUND
# ARPACK_LIBRARY
#

find_library(ARPACK_LIBRARY
  NAMES "arpack"
  PATH_SUFFIXES "lib" "lib32" "lib64"
)

# Handle the QUIETLY and REQUIRED arguments and set the HPDDM_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ARPACK DEFAULT_MSG
                                  ARPACK_LIBRARY)

mark_as_advanced(ARPACK_LIBRARY)

if (ARPACK_FOUND)
    set(ARPACK_LIBRARIES ${ARPACK_LIBRARY} )
endif()