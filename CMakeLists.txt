cmake_minimum_required(VERSION 3.10)
project(xor_cpp)

set(CMAKE_CXX_STANDARD 17)

# add_executable(myproject
#     main.cpp
#     # helper.cpp
#     # math_utils.cpp
# )

file(GLOB SOURCES *.cpp)
add_executable(xor_cpp ${SOURCES})


add_custom_target(run
    COMMAND xor_cpp
    DEPENDS xor_cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)