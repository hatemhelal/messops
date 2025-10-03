cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# --- Project Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
include(GNUInstallDirs)
include(cmake/CPM.cmake)
cpmaddpackage(https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.zip)

cpmaddpackage(
  NAME
  libint2
  VERSION
  2.11.1
  URL
  "https://github.com/evaleev/libint/releases/download/v2.11.1/libint-2.11.1-mpqc4.tgz"
  OPTIONS
  "LIBINT_USE_BUNDLED_BOOST ON"
  "REQUIRE_CXX_API ON"
  "LIBINT2_BUILD_SHARED_AND_STATIC_LIBS ON"
  "LIBINT2_REALTYPE double")

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core NB_STATIC src/core.cpp)

# WORKAROUND: libint2 v2.11.1 does not provide a target for unpacking its
# bundled Boost headers, leading to a build race condition. We define a custom
# target here to unpack and establish an explicit dependency.
add_custom_command(
  OUTPUT ${libint2_BINARY_DIR}/include/libint2/boost/preprocessor.hpp
  COMMAND ${CMAKE_COMMAND} -E tar xf ${libint2_SOURCE_DIR}/external/boost.tar.gz
  WORKING_DIRECTORY ${libint2_BINARY_DIR}/include/libint2
  COMMENT "Unpacking bundled Boost.Preprocessor for libint2"
  VERBATIM)
add_custom_target(
  unpack_libint_boost
  DEPENDS ${libint2_BINARY_DIR}/include/libint2/boost/preprocessor.hpp)

# Make the _core module depend on our custom unpacking target.
add_dependencies(_core unpack_libint_boost)
target_link_libraries(_core PRIVATE libint2 Eigen3::Eigen)

# --- Installation ---
# Set RPATH so the module can find libint2.so. In the build tree, it will look
# in the libint2 build directory. After installation, it will look in the same
# directory as the module itself. We use @rpath for the library reference and
# set the install RPATH to @loader_path (macOS) or $ORIGIN (Linux).
set(CMAKE_MACOSX_RPATH ON)
if(APPLE)
  set(INSTALL_RPATH_DIR "@loader_path")
else()
  set(INSTALL_RPATH_DIR "$ORIGIN")
endif()
set_target_properties(_core PROPERTIES BUILD_RPATH "${libint2_BINARY_DIR}/src"
                                       INSTALL_RPATH "${INSTALL_RPATH_DIR}")

# Install the _core module and its libint2 dependency into the package
# directory.
install(TARGETS _core LIBRARY DESTINATION messops)
install(TARGETS libint2 LIBRARY DESTINATION messops)
