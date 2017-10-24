#
# Try to find HPDDM library and include path.
# Once done this will define
#
# SUITESPARSE_FOUND
# SUITESPARSE_INCLUDE_DIRS
# SUITESPARSE_LIBRARIES

FIND_PATH(
  SUITESPARSE_INCLUDE_DIR
  NAMES umfpack.h
    )
mark_as_advanced(SUITESPARSE_INCLUDE_DIR)
set(SUITESPARSE_INCLUDE_DIRS ${SUITESPARSE_INCLUDE_DIR})

# create list of libs to find
set(SUITESPARSE_LIBS_to_find "umfpack")
list(APPEND SUITESPARSE_LIBS_to_find "cholmod")

set(SUITESPARSE_LIBRARIES "")

foreach(suitesparse_lib ${SUITESPARSE_LIBS_to_find})
  SET(SUITESPARSE_${suitesparse_lib}_LIBRARY "SUITESPARSE_${suitesparse_lib}_LIBRARY-NOTFOUND")
  FIND_LIBRARY(SUITESPARSE_${suitesparse_lib}_LIBRARY
    NAMES ${suitesparse_lib}
  )
  mark_as_advanced(SUITESPARSE_${suitesparse_lib}_LIBRARY)
  list(APPEND SUITESPARSE_LIBRARIES ${SUITESPARSE_${suitesparse_lib}_LIBRARY})
endforeach ()

# Handle the QUIETLY and REQUIRED arguments and set the HPDDM_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SUITESPARSE DEFAULT_MSG
SUITESPARSE_INCLUDE_DIRS SUITESPARSE_LIBRARIES)
