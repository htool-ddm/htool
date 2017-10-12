#
# Try to find HPDDM library and include path.
# Once done this will define
#
# MUMPS_FOUND
# MUMPS_INCLUDE_DIRS
# MUMPS_LIBRARIES

FIND_PATH(
  MUMPS_INCLUDE_DIR
  NAMES mumps_compat.h
    )
mark_as_advanced(MUMPS_INCLUDE_DIR)
set(MUMPS_INCLUDE_DIRS ${MUMPS_INCLUDE_DIR})
# create list of libs to find
set(MUMPS_LIBS_to_find "mumps_common")
list(APPEND MUMPS_LIBS_to_find "smumps")
list(APPEND MUMPS_LIBS_to_find "dmumps")
list(APPEND MUMPS_LIBS_to_find "cmumps")
list(APPEND MUMPS_LIBS_to_find "zmumps")
# list(APPEND MUMPS_LIBS_to_find "mpiseq")

set(MUMPS_LIBRARIES "")

foreach(mumps_lib ${MUMPS_LIBS_to_find})
  SET(MUMPS_${mumps_lib}_LIBRARY "MUMPS_${mumps_lib}_LIBRARY-NOTFOUND")
  FIND_LIBRARY(MUMPS_${mumps_lib}_LIBRARY
    NAMES ${mumps_lib}
  )
  mark_as_advanced(MUMPS_${mumps_lib}_LIBRARY)
  list(APPEND MUMPS_LIBRARIES ${MUMPS_${mumps_lib}_LIBRARY})
endforeach ()

# Handle the QUIETLY and REQUIRED arguments and set the HPDDM_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MUMPS DEFAULT_MSG
MUMPS_INCLUDE_DIRS MUMPS_LIBRARIES)
