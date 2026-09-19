cmake_minimum_required(VERSION 3.20)

project(scalergrad
    VERSION 0.1.0
    LANGUAGES CXX
)

add_library(scalergrad INTERFACE)

target_include_directories(scalergrad
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(scalergrad
    INTERFACE
        cxx_std_17
)