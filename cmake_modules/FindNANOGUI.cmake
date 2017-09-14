#
# Try to find NANOGUI library and include path.
# Once done this will define
#
# NANOGUI_FOUND
# NANOGUI_INCLUDE_DIR
# NANOGUI_LIBRARY
#


FIND_PATH(NANOGUI_INCLUDE_DIR nanogui/nanogui.h
  PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/nanogui/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/nanogui/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/nanogui/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../nanogui/include
    )

FIND_LIBRARY( NANOGUI_LIBRARY NAMES nanogui
  PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/nanogui/build
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/nanogui/build
    ${CMAKE_CURRENT_SOURCE_DIR}/external/nanogui/build
    ${CMAKE_CURRENT_SOURCE_DIR}/../nanogui/build
    ${PROJECT_SOURCE_DIR}/../../external/glfw/lib/x64
    ${PROJECT_SOURCE_DIR}/../external/glfw/lib/x64
    ${PROJECT_SOURCE_DIR}/external/glfw/lib/x64
    PATH_SUFFIXES
    a
    lib64
    lib
)


IF (NANOGUI_INCLUDE_DIR)
  SET(NANOGUI_INCLUDE_DIRS
         ${NANOGUI_INCLUDE_DIR}
         ${NANOGUI_INCLUDE_DIR}/../ext/nanovg/src
         ${NANOGUI_INCLUDE_DIR}/../ext/glfw/include
         )
ENDIF ()

# Handle the QUIETLY and REQUIRED arguments and set the HPDDM_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NANOGUI DEFAULT_MSG
                                  NANOGUI_INCLUDE_DIR
                                  NANOGUI_LIBRARY)

mark_as_advanced(NANOGUI_INCLUDE_DIR)
