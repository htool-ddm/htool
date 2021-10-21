# Git
find_package(Git)
if(Git_FOUND)
    message(STATUS "Git found: ${GIT_EXECUTABLE}")
else()
    message(STATUS "Git not found!")
endif()

function(check_version_number CODE_VERSION_FILE CODE_VARIABLE_VERSION)
    # Git version number
    if(Git_FOUND)
        set(git_version_number "unknown")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
            WORKING_DIRECTORY "${local_dir}"
            OUTPUT_VARIABLE git_version_number
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()

    # source version number
    set(VERSION_FILE)
    if(EXISTS ${CODE_VERSION_FILE})
        set(VERSION_FILE ${CODE_VERSION_FILE})
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${CODE_VERSION_FILE})
        set(VERSION_FILE ${VERSION_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/${CODE_VERSION_FILE})
    else()
        message(FATAL_ERROR "Source file with version number not found")
    endif()

    file(READ ${VERSION_FILE} ver)
    string(REGEX MATCH "${CODE_VARIABLE_VERSION} \"([0-9]+.[0-9]+.[0-9]+)\"" _ ${ver})
    set(code_version_number ${CMAKE_MATCH_1})

    # Check version number: error if code unconsistent
    if(NOT "${code_version_number}" STREQUAL "${CMAKE_PROJECT_VERSION}")
        message(FATAL_ERROR "Inconsistent version number:\n* Source code version number: ${code_version_number}\n* CMake version number: ${CMAKE_PROJECT_VERSION}\n")
    endif()
    # Check version number: warning if git tags inconsistent
    if(NOT "${git_version_number}" STREQUAL "v${CMAKE_PROJECT_VERSION}")
        message(WARNING "Inconsistent version number:\n* GIT last tag: ${git_version_number}\n* Source code version number: ${code_version_number}\n* CMake version number: ${CMAKE_PROJECT_VERSION}\n")
    endif()
endfunction()
