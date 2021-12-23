find_package(Doxygen)
if(NOT DOXYGEN_FOUND)
    message("Doxygen not found, I will not generate/install the documentation")
else()
    # set input and output files
    message(${CMAKE_CURRENT_SOURCE_DIR})
    message(${CMAKE_CURRENT_BINARY_DIR})
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    # note the option ALL which allows to build the docs together with the application
    add_custom_target(
        doc ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

    #install(DIRECTORY ${CMAKE_BINARY_DIR}/doc/html
    #        DESTINATION share/doc/${CMAKE_PROJECT_NAME})

endif()
