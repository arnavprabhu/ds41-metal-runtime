if (NOT APPLE)
    return()
endif()

# standalone Metal latency microbenchmark (no ggml / llama dependency)
find_library(DS41_LAT_FOUNDATION Foundation REQUIRED)
find_library(DS41_LAT_METAL      Metal      REQUIRED)

set(TARGET llama-ds41-latency)
add_executable(${TARGET} ds41-latency.m)
target_link_libraries(${TARGET} PRIVATE ${DS41_LAT_FOUNDATION} ${DS41_LAT_METAL})
