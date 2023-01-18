cmake_minimum_required(VERSION 3.0.0)
project(MlirBytecodeParse VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(MLIRBC_VERBOSE_ERRORS "Enable verbose error reporting" ON)
if(MLIRBC_VERBOSE_ERRORS)
    message(STATUS "Building with verbose error reporting")
    add_definitions(-DMLIRBC_VERBOSE_ERROR)
endif(MLIRBC_VERBOSE_ERRORS)
option(MLIRBC_DEBUG "Enable debug reporting" OFF)
if(MLIRBC_DEBUG)
    message(STATUS "Building with debugging messages on")
    add_definitions(-DMLIRBC_DEBUG)
endif(MLIRBC_DEBUG)
set(MLIR_DIR "" CACHE PATH "MLIR install dir.")
set(LLVM_USE_SANITIZER "" CACHE STRING
    "Define the sanitizer used to build binaries and tests.")

set (CMAKE_C_STANDARD 17)

# If MLIR dir is set, build support libs.
if(IS_DIRECTORY ${MLIR_DIR})
    message(STATUS "Building with MLIR dir `${MLIR_DIR}`")
    find_package(MLIR REQUIRED CONFIG)

    message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

    set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
    set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
    set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)

    include_directories(${LLVM_INCLUDE_DIRS})
    include_directories(${MLIR_INCLUDE_DIRS})
    link_directories(${LLVM_BUILD_LIBRARY_DIR})
    add_definitions(${LLVM_DEFINITIONS})

    # add_subdirectory(tblgen)
else()
    message(STATUS "Building without MLIR (MLIR_DIR `${MLIR_DIR}` not directory)")
    # Simple sanitizer support without LLVM/repeating HandleLLVMOptions that
    # uses same interface.
    if(LLVM_USE_SANITIZER)
        if (LLVM_USE_SANITIZER STREQUAL "Address")
          append_common_sanitizer_flags()
          append("-fsanitize=address" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
        else()
          message(FATAL_ERROR "Unsupported value of LLVM_USE_SANITIZER: ${LLVM_USE_SANITIZER}")
        endif()
    endif()
endif()

# Address
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${PROJECT_BINARY_DIR})
add_subdirectory(examples)

include(CTest)
enable_testing()
