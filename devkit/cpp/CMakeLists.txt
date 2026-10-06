# Build del devkit ufficiale KITTI object detection.
#
# Il CMakeLists.txt originale non linka Boost, senza il quale non compila:
# evaluate_object.cpp usa boost::geometry e boost::numeric::ublas.
#
#   brew install boost cmake
#   cmake -S devkit/cpp -B devkit/build && cmake --build devkit/build

cmake_minimum_required(VERSION 3.16)
project(devkit_object CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED)

add_executable(evaluate_object evaluate_object.cpp)

target_include_directories(evaluate_object PRIVATE ${Boost_INCLUDE_DIRS})
target_compile_options(evaluate_object PRIVATE -O2 -Wno-deprecated-declarations)

# Boost geometriche e numeriche sono header-only: basta la direzione degli include.