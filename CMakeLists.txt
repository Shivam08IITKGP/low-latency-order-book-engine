cmake_minimum_required(VERSION 3.16)
project(OrderBook CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

# Aggressive optimisation flags
add_compile_options(-O3 -march=native -mtune=native -fno-omit-frame-pointer)

# Link-Time Optimisation: lets the compiler inline across .cpp boundaries,
# recovering any performance lost vs the original single-translation-unit build.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

add_executable(orderbook
    main.cpp
    timing.cpp
    cpu_utils.cpp
    messages.cpp
    order_book.cpp
    threads.cpp
    latency.cpp
)

target_link_libraries(orderbook PRIVATE pthread)
target_include_directories(orderbook PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(run
    COMMAND orderbook
    DEPENDS orderbook
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running order book engine"
)