//===-- RawBytecodeTypes.h - MLIR bytecode basic types ------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

#ifndef MLIRBC_BYTECODETYPES_H
#define MLIRBC_BYTECODETYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t MlirBytecodeSize;

// Handles for the different types.
// Note: These should probably all be explicit structs to go beyond compiler
// warnings and to errors. Currently this isn't done as the type is mostly used
// for documentation, warnings help and avoids needing multiple variant of the
// same read function.
struct MlirBytecodeHandle {
  MlirBytecodeSize id;
};
typedef struct MlirBytecodeHandle MlirBytecodeHandle;
typedef MlirBytecodeHandle MlirBytecodeAttrHandle;
typedef MlirBytecodeHandle MlirBytecodeDialectHandle;
typedef MlirBytecodeHandle MlirBytecodeOpHandle;
typedef MlirBytecodeHandle MlirBytecodeStringHandle;
typedef MlirBytecodeHandle MlirBytecodeTypeHandle;
typedef MlirBytecodeHandle MlirBytecodeLocHandle;

// Handle to operation state.
struct MlirBytecodeOperationStateHandle {
  void *state;
};
typedef struct MlirBytecodeOperationStateHandle
    MlirBytecodeOperationStateHandle;

// Reference to section of memory.
// Does not own the underlying string. This is equivalent to llvm::StringRef.
struct MlirBytecodeBytesRef {
  // Pointer to the start memory address.
  const uint8_t *data;
  // Length of the fragment.
  size_t length;
};
typedef struct MlirBytecodeBytesRef MlirBytecodeBytesRef;

struct MlirBytecodeOpRef {
  // Handle to op dialect.
  MlirBytecodeDialectHandle dialect;
  // Handle to op.
  MlirBytecodeStringHandle op;
};
typedef struct MlirBytecodeOpRef MlirBytecodeOpRef;

// MLIR file structure.
typedef struct MlirBytecodeFile MlirBytecodeFile;

// MLIR bytecode stream.
struct MlirBytecodeStream {
  const uint8_t *start;
  const uint8_t *pos;
  const uint8_t *end;
};
typedef struct MlirBytecodeStream MlirBytecodeStream;

// Handle iterator.
struct MlirBytecodeHandleIterator {
  // Stream over block args.
  MlirBytecodeStream stream;

  // Number of handles in stream.
  const MlirBytecodeSize count;
};
typedef struct MlirBytecodeHandleIterator MlirBytecodeHandleIterator;

enum MlirBytecodeAsmResourceEntryKind {
  /// A blob of data with an accompanying alignment.
  kMlirBytecodeResourceEntryBlob,
  /// A boolean value.
  kMlirBytecodeResourceEntryBool,
  /// A string value.
  kMlirBytecodeResourceEntryString,
};
typedef enum MlirBytecodeAsmResourceEntryKind MlirBytecodeAsmResourceEntryKind;

#ifdef __cplusplus
}
#endif

#endif // MLIRBC_BYTECODETYPES_H
