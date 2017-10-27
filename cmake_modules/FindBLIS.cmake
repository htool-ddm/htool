#
# Try to find HPDDM library and include path.
# Once done this will define
#
# BLIS_FOUND
# BLIS_LIBRARIES

# create list of libs to find
set(BLIS_LIBS_to_find "blis")
list(APPEND BLIS_LIBS_to_find "memkind")
list(APPEND BLIS_LIBS_to_find "autohbw")

set(BLIS_LIBRARIES "")

foreach(BLIS_lib ${BLIS_LIBS_to_find})
  SET(BLIS_${BLIS_lib}_LIBRARY "BLIS_${BLIS_lib}_LIBRARY-NOTFOUND")
  FIND_LIBRARY(BLIS_${BLIS_lib}_LIBRARY
    NAMES ${BLIS_lib}
    PATHS ${BLIS_CHECK_LIBRARY_DIRS}
  )
  mark_as_advanced(BLIS_${BLIS_lib}_LIBRARY)
  list(APPEND BLIS_LIBRARIES ${BLIS_${BLIS_lib}_LIBRARY})
endforeach ()


# Handle the QUIETLY and REQUIRED arguments and set the HPDDM_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLIS DEFAULT_MSG BLIS_LIBRARIES)
