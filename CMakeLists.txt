cmake_minimum_required(VERSION 3.20)

project(scalergrad
    VERSION 0.1.0
    LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(scalergrad INTERFACE)
add_library(scalergrad::scalergrad ALIAS scalergrad)

target_include_directories(scalergrad
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(scalergrad
    INTERFACE
        cxx_std_17
)

# ---- Install headers ----
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/scalergrad
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# ---- Install target + export ----
install(
    TARGETS scalergrad
    EXPORT scalergradTargets
)

install(
    EXPORT scalergradTargets
    FILE scalergradTargets.cmake
    NAMESPACE scalergrad::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scalergrad
)

# ---- Version + config files so find_package(scalergrad) works ----
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/scalergradConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scalergradConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/scalergradConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scalergrad
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/scalergradConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/scalergradConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scalergrad
)