#
# Try to find HPDDM library and include path.
# Once done this will define
#
# BLIS_FOUND
# BLIS_LIBRARIES


# create list of libs to find
FIND_LIBRARY(BLIS_LIBRARIES
  NAMES blis
)


# Handle the QUIETLY and REQUIRED arguments and set the HPDDM_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLIS DEFAULT_MSG BLIS_LIBRARIES)
