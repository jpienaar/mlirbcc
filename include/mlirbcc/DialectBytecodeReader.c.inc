//===-- DialectBytecodeReader.c.inc - Interface dialects use ------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// MLIR C bytecode reader for dialects.
//===----------------------------------------------------------------------===//

#ifndef MLIRBC_DIALECTBYTECODEREADER_H
#define MLIRBC_DIALECTBYTECODEREADER_H

#include "mlirbcc/BytecodeTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Reader struct dialect attribute & type parsing should use.
typedef struct MlirBytecodeDialectReader MlirBytecodeDialectReader;

typedef struct MlirBytecodeAPInt MlirBytecodeAPInt;

static MlirBytecodeStatus
MlirBytecodeDialectReaderReadAttribute(MlirBytecodeDialectReader *reader,
                                       MlirBytecodeAttrHandle *result);

static MlirBytecodeStatus
MlirBytecodeDialectReaderReadType(MlirBytecodeDialectReader *reader,
                                  MlirBytecodeTypeHandle *result);

static MlirBytecodeStatus
MlirBytecodeDialectReaderReadVarInt(MlirBytecodeDialectReader *reader,
                                    uint64_t *result);

static MlirBytecodeStatus
MlirBytecodeDialectReaderReadSignedVarInt(MlirBytecodeDialectReader *reader,
                                          int64_t *result);

// TODO: Should allocator be a hook?
static MlirBytecodeStatus MlirBytecodeDialectReaderReadAPIntWithKnownWidth(
    MlirBytecodeDialectReader *reader, unsigned bitWidth,
    void *(*alloc)(size_t size), MlirBytecodeAPInt *);

static MlirBytecodeStatus
MlirBytecodeDialectReaderReadString(MlirBytecodeDialectReader *reader,
                                    MlirBytecodeBytesRef *result);

static MlirBytecodeStatus
MlirBytecodeDialectReaderReadBlob(MlirBytecodeDialectReader *reader,
                                  MlirBytecodeBytesRef *result);

// FailureOr<APFloat>
// MlirBytecodeDialectReaderReadAPFloatWithKnownSemantics(MlirBytecodeDialectReader*
// reader, const llvm::fltSemantics &semantics) override;

struct MlirBytecodeAPInt {
  // If bitWidth <= 64 then value is populated, else
  union {
    int64_t value;
    uint64_t *data;
  } U;

  unsigned bitWidth;
};

#ifdef __cplusplus
}
#endif

struct MlirBytecodeDialectReader  {
  void* callerState;
  MlirBytecodeStream* stream;
};

MlirBytecodeStatus mlirBytecodeParseAPIntWithKnownWidth(
    MlirBytecodeDialectReader *reader, unsigned bitWidth,
    void *(*allocFn)(size_t size), MlirBytecodeAPInt *result) {
  result->bitWidth = bitWidth;

  MlirBytecodeStatus ret;
  // Small values are encoded using a single byte.
  if (bitWidth <= 8) {
    uint8_t value;
    ret = mlirBytecodeParseByte(reader->callerState, reader->stream, &value);
    if (!mlirBytecodeSucceeded(ret))
      return ret;
    return ret;

    result->U.value = value;
    return mlirBytecodeSuccess();
  }

  // Large values up to 64 bits are encoded using a single varint.
  if (bitWidth <= 64) {
    ret = mlirBytecodeParseSignedVarInt(reader->callerState, reader->stream, &result->U.value);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
    return mlirBytecodeSuccess();
  }

  // Otherwise, for really big values we encode the array of active words in
  // the value.

  // Check if allocation function was provided.
  if (!allocFn) return mlirBytecodeUnhandled();

  uint64_t numActiveWords;
  ret = mlirBytecodeParseVarInt(reader->callerState, reader->stream, &numActiveWords);
  if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
    return ret;

  result->U.data = (uint64_t *)allocFn(numActiveWords * sizeof(uint64_t));
  if (!result->U.data)
    return mlirBytecodeFailure();

  for (uint64_t i = 0; i < numActiveWords; ++i) {
    ret = mlirBytecodeParseSignedVarInt(reader->callerState, reader->stream,
                                        (int64_t *)&result->U.data[i]);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
  }
  return mlirBytecodeSuccess();
}
#endif // MLIRBC_DIALECTBYTECODEREADER_H