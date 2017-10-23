#
# Try to find HPDDM library and include path.
# Once done this will define
#
# HTOOL_FOUND
# HTOOL_INCLUDE_DIRS
#

FIND_PATH(
  HTOOL_INCLUDE_DIR
  NAMES htool/htool.hpp
  PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/../htool/include
    )

# Handle the QUIETLY and REQUIRED arguments and set the HPDDM_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HTOOL DEFAULT_MSG
                                  HTOOL_INCLUDE_DIR)

mark_as_advanced(HTOOL_INCLUDE_DIR)

if (HTOOL_FOUND)
    set(HTOOL_INCLUDE_DIRS ${HTOOL_INCLUDE_DIR} )
endif()
