vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO <your-github-username>/scalergrad
    REF "v${VERSION}"
    SHA512 0 # placeholder — vcpkg will print the real SHA512 the first time you run `vcpkg install`; paste it back here
    HEAD_REF main
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(
    PACKAGE_NAME scalergrad
    CONFIG_PATH lib/cmake/scalergrad
)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug")

file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/scalergrad" RENAME copyright)