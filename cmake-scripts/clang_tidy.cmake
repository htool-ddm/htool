#
# clang-tidy
#
# Deliberately NOT wired via CMAKE_CXX_CLANG_TIDY: that hooks clang-tidy into
# every compiler invocation of every target, which is expensive for a
# header-heavy, template-heavy library and multiplies badly across the
# project's many build configurations. Instead this mirrors the opt-in
# clang_format()/cmake_format() pattern already used in formatting.cmake:
# a target you run on demand (cmake --build <dir> --target tidy), not part
# of ALL.
#
# Requires CMAKE_EXPORT_COMPILE_COMMANDS to be ON (already set in the main
# CMakeLists.txt) so clang-tidy can find compile flags via -p.
find_program(CLANG_TIDY_EXE "clang-tidy")
mark_as_advanced(FORCE CLANG_TIDY_EXE)
if(CLANG_TIDY_EXE)
  message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
else()
  message(STATUS "clang-tidy not found!")
endif()

# Generates a target running clang-tidy over the given list of translation
# units (.cpp files -- pass files that actually have a compile command, e.g.
# from tests/ or examples/, not headers directly; headers pulled in by those
# TUs are still checked, filtered by HeaderFilterRegex in .clang-tidy).
#
# ~~~
# Required:
# TARGET_NAME - The name of the target to create.
#
# Optional: ARGN - The list of files to analyze. Relative and absolute paths
# are accepted.
# ~~~
function(clang_tidy TARGET_NAME)
  if(CLANG_TIDY_EXE)
    set(TIDY_FILES)
    foreach(item IN LISTS ARGN)
      if(EXISTS ${item})
        set(TIDY_FILES ${TIDY_FILES} ${item})
      elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${item})
        set(TIDY_FILES ${TIDY_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/${item})
      endif()
    endforeach()

    if(TIDY_FILES)
      add_custom_target(
        ${TARGET_NAME}
        COMMAND ${CLANG_TIDY_EXE} -p ${CMAKE_BINARY_DIR} ${TIDY_FILES}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running clang-tidy"
        VERBATIM)

      if(NOT TARGET tidy)
        add_custom_target(tidy)
      endif()

      add_dependencies(tidy ${TARGET_NAME})
    endif()
  endif()
endfunction()
