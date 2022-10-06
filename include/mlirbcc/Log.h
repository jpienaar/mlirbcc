//===-- Log.h - MLIR bytecode C logging helpers -------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Helpers for debugging logging and error return method.
//===----------------------------------------------------------------------===//

#ifndef MLIRBC_LOG_H
#define MLIRBC_LOG_H

#include "mlirbcc/Status.h"

/// Helper function to emit error and return failure status.
extern MlirBytecodeStatus mlirBytecodeEmitErrorImpl(const char *fmt, ...);

/// Helper function to emit debugging info.
extern void mlirBytecodeEmitDebugImpl(const char *fmt, ...);

// Macros to allow capturing file and line from which invoked.
#ifdef MLIRBC_DEBUG
#define mlirBytecodeEmitDebug(fmt, ...)                                        \
  mlirBytecodeEmitDebugImpl(state, "%s:%d " fmt, __FILE__, __LINE__,           \
                            __VA_ARGS__)
#else
#define mlirBytecodeEmitDebug(...)                                             \
  while (0)                                                                    \
    ;
#endif

#ifdef MLIRBC_VERBOSE_ERRORS
#define mlirBytecodeEmitError(fmt, ...)                                        \
  mlirBytecodeEmitErrorImpl(fmt, __VA_ARGS__)
#else
#define mlirBytecodeEmitError(...) mlirBytecodeFailure()
#endif
#endif // MLIRBC_LOG_H
