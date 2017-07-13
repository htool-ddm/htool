#
# Try to find HPDDM library and include path.
# Once done this will define
#
# HPDDM_FOUND
# HPDDM_INCLUDE_DIR
#

FIND_PATH(
  HPDDM_INCLUDE_DIR HPDDM.hpp
  PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/../hpddm/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/hpddm/include
    )


# Handle the QUIETLY and REQUIRED arguments and set the HPDDM_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HPDDM DEFAULT_MSG
                                  HPDDM_INCLUDE_DIR)

mark_as_advanced(HPDDM_INCLUDE_DIR)
