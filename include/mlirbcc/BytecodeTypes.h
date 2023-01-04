//===-- RawBytecodeTypes.h - MLIR bytecode basic types ------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

#ifndef MLIRBC_RAWBYTECODETYPES_H
#define MLIRBC_RAWBYTECODETYPES_H

#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Handles for the different types.
// Note: These should probably all be explicit structs to go beyond compiler
// warnings and to errors. Currently this isn't done as the type is mostly used
// for documentation, warnings help and avoids needing multiple variant of the
// same read function.
typedef struct {
  uint64_t id;
} MlirBytecodeHandle;
typedef uint64_t MlirBytecodeSize;
typedef MlirBytecodeHandle MlirBytecodeAttrHandle;
typedef MlirBytecodeHandle MlirBytecodeDialectHandle;
typedef MlirBytecodeHandle MlirBytecodeOpHandle;
typedef MlirBytecodeHandle MlirBytecodeStringHandle;
typedef MlirBytecodeHandle MlirBytecodeTypeHandle;
typedef MlirBytecodeHandle MlirBytecodeLocHandle;

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
  MlirBytecodeOpHandle op;
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

typedef struct MlirBytecodeHandleIterator MlirBlockArgHandleIterator;
typedef struct MlirBytecodeHandleIterator MlirBytecodeStringIterator;
typedef struct MlirBytecodeHandleIterator MlirBytecodeDialectNameRange;

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

#endif // MLIRBC_RAWBYTECODETYPES_H
