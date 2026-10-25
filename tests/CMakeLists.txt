llama_add_compile_flags()

# Functions used by test-backend-ops.cpp
function(llama_build source)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs LINK)
    cmake_parse_arguments(LLAMA_BUILD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_BUILD_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_BUILD_LINK)
        set(LLAMA_BUILD_LINK llama-common)
    endif()

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_BUILD_LINK})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

# Build test-backend-ops only
llama_build(test-backend-ops.cpp LINK ggml)
