#=============================================================================#
#============================== Project ======================================#
#=============================================================================#
cmake_minimum_required(VERSION 3.10)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    project(
        Htool
        VERSION 0.9.0
        LANGUAGES CXX)
else()
    project(
        Htool
        VERSION 0.9.0
        DESCRIPTION "A header only c++ library that provides Hierarchical matrices."
        HOMEPAGE_URL "https://github.com/htool-ddm/htool"
        LANGUAGES CXX)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # To set default CMAKE_BUILD_TYPE
    set(default_build_type "Release")
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
        set(CMAKE_BUILD_TYPE
            "${default_build_type}"
            CACHE STRING "Choose the type of build." FORCE)
    endif()

    if(${CMAKE_BUILD_TYPE} STREQUAL Release_native)
        message(STATUS "Setting build type to 'Release_native'.")
        set(CMAKE_BUILD_TYPE Release)
        set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -march=native")
    endif()

    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

    # Version number check
    include(cmake/version.cmake)
    check_version_number("include/htool/htool_version.hpp" "\#define HTOOL_VERSION_MAJOR" "\#define HTOOL_VERSION_MINOR" "\#define HTOOL_VERSION_SUBMINOR")

    # Sanitizers
    include(cmake-scripts/sanitizers.cmake)

    # Formatting
    include(cmake-scripts/formatting.cmake)
    file(
        GLOB_RECURSE
        ALL_CODE_FILES
        ${PROJECT_SOURCE_DIR}/include/*.[h]pp
        ${PROJECT_SOURCE_DIR}/include/*.[h]
        ${PROJECT_SOURCE_DIR}/examples/*.[ch]pp
        ${PROJECT_SOURCE_DIR}/examples/*.[ch]
        ${PROJECT_SOURCE_DIR}/tests/*.[ch]pp
        ${PROJECT_SOURCE_DIR}/tests/*.[ch])

    clang_format(format ${ALL_CODE_FILES})
    file(GLOB_RECURSE CMAKE_FILES ${PROJECT_SOURCE_DIR}/CMakeLists.txt)
    cmake_format(cmake_format ${CMAKE_FILES})

    # Files to do find_package for some module
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_modules")

    # Information about compilation exported
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    # Let's nicely support folders in IDE's
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    # Testing only available if this is the main app
    # Note this needs to be done in the main CMakeLists
    # since it calls enable_testing, which must be in the
    # main CMakeLists.
    include(CTest)

endif()

# Options
option(HTOOL_WITH_EXAMPLES "Build htool examples ?" ON)
option(HTOOL_WITH_DOC "Build documentation" ON)
option(HTOOL_WITH_STRICT_TESTS "Add -Werror to the tests" OFF)

#=============================================================================#
#========================== External Libraries ===============================#
#=============================================================================#
# MPI
find_package(MPI REQUIRED)
message(STATUS "MPI libraries:" "${MPI_LIBRARIES}")
message(STATUS "MPI include files:" "${MPI_INCLUDE_PATH}")
message(STATUS "Tests run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
separate_arguments(MPIEXEC_PREFLAGS) # to support multi flags

# OPENMP
find_package(OpenMP)

# BLAS
find_package(BLAS REQUIRED)
message(STATUS "Blas implementation:" "${BLAS_LIBRARIES}")

# LAPACK
find_package(LAPACK)
message(STATUS "Lapack implementation:" "${LAPACK_LIBRARIES}")

# HPDDM
find_package(HPDDM)

#=============================================================================#
#========================== Libraries ========================================#
#=============================================================================#
#=== HTOOL as header only library
add_library(htool INTERFACE)
target_include_directories(htool INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include> ${HPDDM_INCLUDE_DIRS} ${MKL_INC_DIR})
target_link_libraries(htool INTERFACE MPI::MPI_CXX BLAS::BLAS LAPACK::LAPACK)
if(OpenMP_CXX_FOUND)
    target_link_libraries(htool INTERFACE OpenMP::OpenMP_CXX)
endif()
target_compile_features(htool INTERFACE cxx_std_17)

if("${BLA_VENDOR}" STREQUAL "Intel10_32"
   OR "${BLA_VENDOR}" STREQUAL "Intel10_64lp"
   OR "${BLA_VENDOR}" STREQUAL "Intel10_64lp_seq"
   OR "${BLA_VENDOR}" STREQUAL "Intel10_64ilp"
   OR "${BLA_VENDOR}" STREQUAL "Intel10_64ilp_seq"
   OR "${BLA_VENDOR}" STREQUAL "Intel10_64_dyn")
    target_compile_definitions(htool INTERFACE "-DHPDDM_MKL -DHTOOL_MKL")
endif()

# For headers to show in IDE
if(NOT "${CMAKE_VERSION}" VERSION_LESS 3.1)
    target_sources(htool INTERFACE ${htool_include_dir})
endif()

#===
#=============================================================================#
#========================== Installation =====================================#
#=============================================================================#

# Define target to install
install(
    TARGETS htool
    EXPORT ${PROJECT_NAME}Targets
    PUBLIC_HEADER DESTINATION include)

# Build and install CMake helpers
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion)

configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in" "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" INSTALL_DESTINATION lib/cmake/${PROJECT_NAME})

install(
    EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME})

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake" "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" DESTINATION lib/cmake/${PROJECT_NAME})

###### install files
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)

#=============================================================================#
#========================== Repertories ======================================#
#=============================================================================#
# Add examples
if(HTOOL_WITH_EXAMPLES)
    add_custom_target(build-examples)
    add_subdirectory(examples EXCLUDE_FROM_ALL)
endif()

# Add tests

if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME) AND BUILD_TESTING)
    if(CODE_COVERAGE)
        if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            message(STATUS "Code coverage enabled with GNU.")
            target_compile_options(htool INTERFACE -fprofile-arcs -ftest-coverage -fno-elide-constructors -fno-default-inline)
            target_link_libraries(htool INTERFACE gcov)
        elseif(CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message(STATUS "Code coverage enabled with LLVM.")
            target_compile_options(htool INTERFACE -fprofile-instr-generate -fcoverage-mapping)
            target_link_options(htool INTERFACE -fprofile-instr-generate -fcoverage-mapping)
        else()
            message(STATUS "Code coverage not available.")
        endif()
    endif()
    if(HTOOL_WITH_STRICT_TESTS)
        target_compile_options(htool INTERFACE -Werror)
    endif()
    target_compile_options(
        htool
        INTERFACE -Wall
                  -Wextra
                  -Wshadow
                  -Wnon-virtual-dtor
                  -pedantic
                  -Wold-style-cast
                  -Wcast-align
                  -Wunused
                  -Woverloaded-virtual
                  #   -Wconversion
                  #   -Wsign-conversion
                  -Wdouble-promotion
                  -Wno-sign-compare
                  -Wextra-semi
                  -Wzero-as-null-pointer-constant
                  -Wundef)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(htool INTERFACE -Wimplicit-fallthrough -Wextra-semi-stmt -ferror-limit=200)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(
            htool
            INTERFACE -Wmisleading-indentation -Wduplicated-cond -Wduplicated-branches -Wlogical-op
                      #   -Wnull-dereference
                      -Wuseless-cast -fmax-errors=200)
    endif()
    add_custom_target(build-tests)
    add_subdirectory(tests EXCLUDE_FROM_ALL)
endif()

# Add documentation
if(HTOOL_WITH_DOC)
    add_subdirectory(doc EXCLUDE_FROM_ALL)
endif()
