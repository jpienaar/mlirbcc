//===-- Status.h - MLIR status helpers ----------------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Status type and helpers.
//===----------------------------------------------------------------------===//

#ifndef MLIRBC_STATUS_H
#define MLIRBC_STATUS_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

//===----------------------------------------------------------------------===//
// MlirBytecodeStatus.
//===----------------------------------------------------------------------===//

// Value representing the status of parsing. Status is either a success, failure
// unhandled or interrupted. Instances of MlirBytecodeStatus must only be
// inspected using the associated functions.
struct MlirBytecodeStatus {
  int8_t value;
};
typedef struct MlirBytecodeStatus MlirBytecodeStatus;

/// Creates a status representing a success.
inline static MlirBytecodeStatus mlirBytecodeSuccess() {
  MlirBytecodeStatus res = {1};
  return res;
}

/// Creates a status representing a failure.
inline static MlirBytecodeStatus mlirBytecodeFailure() {
  MlirBytecodeStatus res = {0};
  return res;
}

/// Creates a status representing an unhandled case.
inline static MlirBytecodeStatus mlirBytecodeUnhandled() {
  MlirBytecodeStatus res = {2};
  return res;
}

/// Creates a status representing an interrupted iteration.
inline static MlirBytecodeStatus mlirBytecodeIterationInterupt() {
  MlirBytecodeStatus res = {3};
  return res;
}

/// Checks if the given status represents a success.
inline static bool mlirBytecodeSucceeded(MlirBytecodeStatus res) {
  return res.value == mlirBytecodeSuccess().value;
}

/// Checks if the given status represents a failure.
inline static bool mlirBytecodeFailed(MlirBytecodeStatus res) {
  return res.value == mlirBytecodeFailure().value;
}

/// Checks if the given status represents a failure.
inline static bool mlirBytecodeInterupted(MlirBytecodeStatus res) {
  return res.value == mlirBytecodeIterationInterupt().value;
}

/// Checks if the given status represents a handled state.
inline static bool mlirBytecodeHandled(MlirBytecodeStatus res) {
  return mlirBytecodeSucceeded(res) || mlirBytecodeFailed(res);
}

#ifdef __cplusplus
}
#endif
#endif // MLIRBC_STATUS_H
