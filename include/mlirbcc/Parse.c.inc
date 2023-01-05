//===-- Parse.c.inc - MLIR bytecode C parser ----------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// MLIR C bytecode event-driven parser.
//===----------------------------------------------------------------------===//

// This file contains the full implementation for an event-based MLIR bytecode
// parser. It is defined as a pure include-only implementation.
//
// Define types and functions required for parsing (see below) and simplest
// entry point is `mlirBytecodeParseFile`.
//
// Callbacks/functions to be implemented in instantiation accept an opaque
// pointer as first argument that gets directly propagated during parsing and
// can be used by instantiation for additional/parse local state capture.
//
// Note: only set implementation define before you include this file in one C or
// C++ file.

// Convention followed in this file is to prefix implementation details with
// mbci_. But there is more to do to differentiate between internal and
// user-facing methods.

#include "mlirbcc/BytecodeTypes.h"

#include <assert.h>
#include <inttypes.h>
#include <stdalign.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(MLIRBC_DEBUG) || defined(MLIRBC_VERBOSE_ERROR)
#include <stdarg.h>
#include <stdio.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// - Configs
#define MlirIRSectionStackMaxDepth 15 // TODO: add dynamic option.

typedef struct MlirBytecodeStatus MlirBytecodeStatus;

//===----------------------------------------------------------------------===//
// Functions required to be defined:
#define MLIRBC_DEF static

/// Called when creating variable to populate operation state in.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStatePush(
    void *callerState, MlirBytecodeOpHandle name, MlirBytecodeLocHandle loc,
    MlirBytecodeOperationStateHandle *opState);

/// Called when finalizing population of operation state.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStatePop(
    void *callerState, MlirBytecodeOperationStateHandle);

/// Called for an operation with attributes.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStateAddAttributes(
    void *callerState, MlirBytecodeOperationStateHandle,
    MlirBytecodeAttrHandle);

/// Called for operation with results.
// Note: This _requires_ that the stream is progressed post the last item before
// this function returns.
// TODO: Consider making these iterators/for each functors instead of exposing
// bytestream to avoid mishaps with progressing stream.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStateAddResultTypes(
    void *callerState, MlirBytecodeOperationStateHandle, MlirBytecodeStream *,
    uint64_t numResults);

/// Called for operation with operands.
// Note: This _requires_ that the stream is progressed post the last item before
// this function returns.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStateAddOperands(
    void *callerState, MlirBytecodeOperationStateHandle, MlirBytecodeStream *,
    uint64_t numOperands);

/// Called for operations with regions.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStateAddRegions(
    void *callerState, MlirBytecodeOperationStateHandle, uint64_t numRegions);

/// Called for operations with successors.
// Note: This _requires_ that the stream is progressed post the last item before
// this function returns.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStateAddSuccessors(
    void *callerState, MlirBytecodeOperationStateHandle, MlirBytecodeStream *,
    uint64_t numSuccessors);

/// Called when entering a region with numBlocks blocks and numValues Values
/// (including values due to block args).
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeRegionPush(void *callerState, MlirBytecodeOperationStateHandle,
                       size_t numBlocks, size_t numValues);

/// Called when entering block.
// Note: This _requires_ that the stream is progressed post the last item before
// this function returns.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStateBlockPush(
    void *callerState, MlirBytecodeOperationStateHandle, MlirBytecodeStream *,
    uint64_t numArgs);

/// Called when exiting the block.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeBlockPop(void *callerState, MlirBytecodeOperationStateHandle);

/// Called when exiting a region.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeRegionPop(void *callerState, MlirBytecodeOperationStateHandle);

/// Called to process an attribute handle and convert it into an attribute.
/// This may result in triggering additional parsing and processing of
/// attributes and types.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeProcessAttribute(void *callerState, MlirBytecodeAttrHandle handle);

/// Called to process a type handle and convert it into an attribute.
/// This may result in triggering additional parsing and processing of
/// attributes and types.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeProcessType(void *callerState, MlirBytecodeTypeHandle handle);

//===----------------------------------------------------------------------===//
// Section parsing entry points.

/// Invoked with dialect and attribute with the attribute's range in memory.
// TODO: This and others currently pass in total number of attributes per
// invocation, consider factoring out to an initialize that sets total size.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeAttrCallBack(
    void *, MlirBytecodeAttrHandle, size_t total, MlirBytecodeDialectHandle,
    MlirBytecodeBytesRef, bool hasCustom);

// Parse and associate attr handle with parsed attribute.
typedef MlirBytecodeStatus (*MlirBytecodeAttrParseCallBack)(
    void *, MlirBytecodeAttrHandle, MlirBytecodeBytesRef, bool hasCustom);

/// Invoked with dialect and type with the type's range in memory.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeTypeCallBack(
    void *, MlirBytecodeTypeHandle, size_t total, MlirBytecodeDialectHandle,
    MlirBytecodeBytesRef, bool hasCustom);

// Parse and associate type handle with parsed type.
typedef MlirBytecodeStatus (*MlirBytecodeTypeParseCallBack)(
    void *, MlirBytecodeTypeHandle, MlirBytecodeBytesRef, bool hasCustom);

/// Iterators over attributes and types, calling MlirBytecodeAttrCallBack and
/// MlirBytecodeTypeCallBack upon encountering Attribute or Type respectively.
/// Returns whether failed.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeForEachAttributeAndType(
    void *callerState, MlirBytecodeFile *mlirFile);

/// Invoked with dialect handle and its corresponding string handle.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeDialectCallBack(void *, MlirBytecodeDialectHandle, size_t /*total*/,
                            MlirBytecodeStringHandle);

/// Invoked with opname with its dialect and string handle corresponding to its
/// name.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeDialectOpCallBack(void *, MlirBytecodeDialectHandle,
                              MlirBytecodeOpHandle, MlirBytecodeStringHandle);

/// Parses the dialect section, invoking MlirBytecodeDialectCallBack upon
/// dialect encountered and MlirBytecodeDialectOpCallBack per operation type in
/// dialect. Returns whether failed.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeParseDialectSection(void *callerState, MlirBytecodeFile *mlirFile);

/// Parse IR section in mlirFile. The block args, operation and region callback
/// are invoked during bytecode in-order walk. Additionally allows for passing
/// in an opaque state.
///
/// The IR section parsing follows the nesting order:
///   op ->* regions ->* blocks
/// The caller is required to keep track of when all operations/blocks in
/// block/region have been processed and so parsing resumes at parent level.
/// Returns whether failed.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeParseIRSection(void *callerState, MlirBytecodeFile *mlirFile);

/// Invoked with groupKey[resourceKey] and either MlirBytecodeStatus, string or
/// blob.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeResourceCallBack(
    void *, MlirBytecodeStringHandle groupKey, MlirBytecodeSize totalGroups,
    MlirBytecodeStringHandle resourceKey, MlirBytecodeAsmResourceEntryKind,
    const MlirBytecodeBytesRef *blob, const uint8_t *MlirBytecodeStatusResource,
    const MlirBytecodeStringHandle *);

/// Parse the resource section, calling MlirBytecodeResourceCallBack upon
/// resources encountered. Returns whether failed.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeForEachResource(void *callerState, MlirBytecodeFile *mlirFile);

/// Invoked with string handle, total number of strings in string section and
/// bytes corresponding to the string per entry in string section.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeStringCallBack(
    void *, MlirBytecodeStringHandle, size_t total, MlirBytecodeBytesRef);

//===----------------------------------------------------------------------===//
// MLIR file parsing methods.

/// Populates the MlirBytecodeFile contents for given in memory bytes.
/// Returns an empty file if population failed.
MLIRBC_DEF MlirBytecodeFile
mlirBytecodePopulateFile(MlirBytecodeBytesRef bytes);

/// Parses the given MLIR file represented in memory `bytes`, calls the
/// appropriate callbacks during parsing. This combines the parsing methods
/// below.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeParseFile(void *callerState,
                                                    MlirBytecodeBytesRef bytes);

/// Returns whether the given MlirBytecodeFile structure is empty.
MLIRBC_DEF bool mlirBytecodeFileEmpty(MlirBytecodeFile *file);

//===----------------------------------------------------------------------===//
// Dialect parsing utility methods.
// Note: These should probably go into their own header.

/// Creates a bytecode stream from section of bytes.
static MlirBytecodeStream mlirBytecodeStreamCreate(MlirBytecodeBytesRef ref);

/// Reset the stream to its head.
static void mlirBytecodeStreamReset(MlirBytecodeStream *stream);

/// Decode the next handle on the stream and increment stream.
static MlirBytecodeStatus mlirBytecodeParseHandle(MlirBytecodeStream *stream,
                                                  MlirBytecodeHandle *result);

/// Skips the next n handles.
static MlirBytecodeStatus mlirBytecodeSkipHandles(MlirBytecodeStream *stream,
                                                  uint64_t n);

/// Decode uint64 on the stream and increment stream.
static MlirBytecodeStatus mlirBytecodeParseVarInt(MlirBytecodeStream *stream,
                                                  uint64_t *result);

/// Decode int64 on the stream and increment stream.
static MlirBytecodeStatus
mlirBytecodeParseSignedVarInt(MlirBytecodeStream *stream, int64_t *result);

/// Parse a variable length encoded integer whose low bit is used to encode a
/// flag, i.e: `(integerValue << 1) | (flag ? 1 : 0)`.
static MlirBytecodeStatus
mlirBytecodeParseVarIntWithFlag(MlirBytecodeStream *stream, uint64_t *result,
                                bool *flag);

/// Parse a null-terminated string.
static MlirBytecodeStatus
mlirBytecodeParseNullTerminatedString(MlirBytecodeStream *stream,
                                      MlirBytecodeBytesRef *str);

//===----------------------------------------------------------------------===//
// They are only used for debugging messages in Parse.h.

// Returns the requested string from the string section.
MLIRBC_DEF MlirBytecodeBytesRef mlirBytecodeGetStringSectionValue(
    void *callerState, const MlirBytecodeFile *mlirFile,
    MlirBytecodeStringHandle idx);

#ifdef __cplusplus
}
#endif

/// Helper function to emit error and return failure status.
extern MlirBytecodeStatus mlirBytecodeEmitErrorImpl(const char *fmt, ...);

/// Helper function to emit debugging info.
extern void mlirBytecodeEmitDebugImpl(const char *file, int line,
                                      const char *fmt, ...);

// Macros to allow capturing file and line from which invoked.
#define mlirBytecodeEmitDebug(...)                                             \
  mlirBytecodeEmitDebugImpl(__FILE__, __LINE__, __VA_ARGS__)

#ifdef MLIRBC_VERBOSE_ERROR
#define mlirBytecodeEmitError(...) mlirBytecodeEmitErrorImpl(__VA_ARGS__)
#else
#define mlirBytecodeEmitError(...) mlirBytecodeFailure()
#endif

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
inline static MlirBytecodeStatus mlirBytecodeIterationInterrupt() {
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
inline static bool mlirBytecodeInterrupted(MlirBytecodeStatus res) {
  return res.value == mlirBytecodeIterationInterrupt().value;
}

/// Checks if the given status represents a handled state.
inline static bool mlirBytecodeHandled(MlirBytecodeStatus res) {
  return mlirBytecodeSucceeded(res) || mlirBytecodeFailed(res);
}

// ----
// Bytecode constants.
// ----

enum ID {
  // This section contains strings referenced within the bytecode.
  mbci_kString = 0,

  // This section contains the dialects referenced within an IR module.
  mbci_kDialect = 1,

  // This section contains the attributes and types referenced within an IR
  // module.
  mbci_kAttrType = 2,

  // This section contains the offsets for the attribute and types within the
  // AttrType section.
  mbci_kAttrTypeOffset = 3,

  // This section contains the list of operations serialized into the bytecode,
  // and their nested regions/operations.
  mbci_kIR = 4,

  // This section contains the resources of the bytecode.
  mbci_kResource = 5,

  // This section contains the offsets of resources within the Resource
  // section.
  mbci_kResourceOffset = 6,

  // The total number of section types.
  mbci_kNumSections = 7,
};

struct MlirBytecodeFile {
  uint64_t version;
  MlirBytecodeBytesRef producer;

  MlirBytecodeBytesRef sectionData[mbci_kNumSections];
};

const char *mbci_sectionIDToString(uint8_t id) {
  const char *arr[] = {
      "String (0)",         //
      "Dialect (1)",        //
      "AttrType (2)",       //
      "AttrTypeOffset (3)", //
      "IR (4)",             //
      "Resource (5)",       //
      "ResourceOffset (6)"  //
  };
  assert(id < sizeof(arr) / sizeof(arr[0]));
  return arr[id];
}

static bool mbci_isSectionOptional(int id) {
  switch (id) {
  case mbci_kResource:
  case mbci_kResourceOffset:
    return true;
  case mbci_kString:
  case mbci_kDialect:
  case mbci_kAttrType:
  case mbci_kAttrTypeOffset:
  case mbci_kIR:
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// Default implementations for verbose logging and debugging
#ifdef MLIRBC_VERBOSE_ERROR
MlirBytecodeStatus mlirBytecodeEmitErrorImpl(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");

  return mlirBytecodeFailure();
}
#else
MlirBytecodeStatus mlirBytecodeEmitErrorImpl(const char *fmt, ...) {
  (void)fmt;
  return mlirBytecodeFailure();
}
#endif

#ifdef MLIRBC_DEBUG
void mlirBytecodeEmitDebugImpl(const char *file, int line, const char *fmt,
                               ...) {
  va_list args;
  fprintf(stderr, "%s:%d: ", file, line);
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");
}
#else
void mlirBytecodeEmitDebugImpl(const char *file, int line, const char *fmt,
                               ...) {
  (void)fmt;
}
#endif

//-----
// Base parsing primitives.
//=====

// Represents simple parser state.
static bool mbci_streamEmpty(MlirBytecodeStream *stream) {
  return stream->pos >= stream->end;
}

static MlirBytecodeStatus mbci_parseByte(MlirBytecodeStream *stream,
                                         uint8_t *val) {
  if (mbci_streamEmpty(stream))
    return mlirBytecodeFailure();
  *val = *stream->pos++;
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus mbci_parseBytes(MlirBytecodeStream *stream,
                                          uint8_t *val, size_t n) {
  for (uint64_t i = 0; i < n; ++i)
    if (mlirBytecodeFailed(mbci_parseByte(stream, val++)))
      return mlirBytecodeFailure();
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus mbci_skipBytes(MlirBytecodeStream *stream, size_t n) {
  stream->pos += n;
  if (stream->pos > stream->end)
    return mlirBytecodeFailure();
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus mbci_skipVarInts(MlirBytecodeStream *stream,
                                           size_t n) {
  for (size_t i = 0; i < n; ++i) {
    uint8_t head;
    if (mlirBytecodeFailed(mbci_parseByte(stream, &head)))
      return mlirBytecodeFailure();
    int numBytes = (head == 0) ? 8 : __builtin_ctz(head);
    if (mlirBytecodeFailed(mbci_skipBytes(stream, numBytes)))
      return mlirBytecodeFailure();
  }
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeParseHandle(MlirBytecodeStream *stream,
                                           MlirBytecodeAttrHandle *result) {
  return mlirBytecodeParseVarInt(stream, &result->id);
}

MlirBytecodeStatus mlirBytecodeSkipHandles(MlirBytecodeStream *stream,
                                           uint64_t n) {
  return mbci_skipVarInts(stream, n);
}

void mlirBytecodeStreamReset(MlirBytecodeStream *stream) {
  stream->pos = stream->start;
}

MlirBytecodeStatus mlirBytecodeParseVarInt(MlirBytecodeStream *stream,
                                           uint64_t *result) {
  // Compute the number of bytes needed to encode the value. Each byte can hold
  // up to 7-bits of data. We only check up to the number of bits we can encode
  // in the first byte (8).
  uint8_t head;
  if (mlirBytecodeFailed(mbci_parseByte(stream, &head)))
    return mlirBytecodeFailure();
  *result = head;
  if ((*result & 1)) {
    *result >>= 1;
    return mlirBytecodeSuccess();
  }

  if (*result == 0) {
    uint8_t *ptr = (uint8_t *)(result);
    if (mlirBytecodeFailed(mbci_parseBytes(stream, ptr, sizeof(*result))))
      return mlirBytecodeFailure();
    return mlirBytecodeSuccess();
  }

  // Parse in the remaining bytes of the value.
  uint32_t numBytes = __builtin_ctz(*result);
  uint8_t *ptr = (uint8_t *)(result) + 1;
  if (mlirBytecodeFailed(mbci_parseBytes(stream, ptr, numBytes)))
    return mlirBytecodeFailure();

  // Shift out the low-order bits that were used to mark how the value was
  // encoded.
  *result >>= (numBytes + 1);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeParseVarIntWithFlag(MlirBytecodeStream *stream,
                                                   uint64_t *result,
                                                   bool *flag) {
  if (mlirBytecodeFailed(mlirBytecodeParseVarInt(stream, result)))
    return mlirBytecodeFailure();
  *flag = *result & 1;
  *result >>= 1;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeParseSignedVarInt(MlirBytecodeStream *stream,
                                                 int64_t *result) {
  if (mlirBytecodeFailed(mlirBytecodeParseVarInt(stream, (uint64_t *)result)))
    return mlirBytecodeFailure();
  *result = (*result >> 1) ^ (~(*result & 1) + 1);
  return mlirBytecodeSuccess();
}

// Align the current reader position to the specified alignment.
static MlirBytecodeStatus mbci_alignTo(MlirBytecodeStream *stream,
                                       uint32_t alignment) {
  bool isPowerOf2 = (alignment != 0) && ((alignment & (alignment - 1)) == 0);
  if (!isPowerOf2)
    return mlirBytecodeEmitError("expected alignment to be a power-of-two");

  // An arbitrary value used to fill alignment padding.
  const uint8_t kAlignmentByte = 0xCB;

  // Shift the reader position to the next alignment boundary.
  while ((uintptr_t)stream->pos & ((uintptr_t)alignment - 1)) {
    uint8_t padding;
    if (mlirBytecodeFailed(mbci_parseByte(stream, &padding)))
      return mlirBytecodeFailure();
    if (padding != kAlignmentByte) {
      return mlirBytecodeEmitError(
          "expected alignment byte (0x%x), but got: '0x%x'", kAlignmentByte,
          padding);
    }
  }

  // TODO: Check that the current data pointer is actually at the expected
  // alignment.
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus
mbci_parseSections(MlirBytecodeStream *stream,
                   MlirBytecodeBytesRef sectionData[mbci_kNumSections]) {
  uint8_t byte;
  uint64_t length;
  if (mlirBytecodeFailed(mbci_parseByte(stream, &byte)) ||
      mlirBytecodeFailed(mlirBytecodeParseVarInt(stream, &length)))
    return mlirBytecodeFailure();
  uint8_t sectionID = byte & 0x7f;

  mlirBytecodeEmitDebug("parsing %d of %lld", sectionID, length);
  if (sectionID >= mbci_kNumSections)
    return mlirBytecodeEmitError("invalid section ID: %d", sectionID);
  if (sectionData[sectionID].data != NULL) {
    return mlirBytecodeEmitError("duplicate top-level section: %s",
                                 mbci_sectionIDToString(sectionID));
  }

  bool isAligned = byte >> 7;
  if (isAligned) {
    uint64_t alignment;
    if (mlirBytecodeFailed(mlirBytecodeParseVarInt(stream, &alignment)) ||
        mlirBytecodeFailed(mbci_alignTo(stream, alignment)))
      return mlirBytecodeFailure();
  }

  sectionData[sectionID].data = stream->pos;
  sectionData[sectionID].length = length;
  return mbci_skipBytes(stream, length);
}

MlirBytecodeStatus
mlirBytecodeParseNullTerminatedString(MlirBytecodeStream *stream,
                                      MlirBytecodeBytesRef *str) {
  const uint8_t *startIt = (const uint8_t *)stream->pos;
  const uint8_t *nulIt =
      (const uint8_t *)memchr(startIt, 0, stream->end - stream->pos);
  if (!nulIt) {
    return mlirBytecodeEmitError(
        "malformed null-terminated string, no null character found");
  }

  str->data = startIt;
  str->length = nulIt - startIt;
  stream->pos += str->length + 1;
  return mlirBytecodeSuccess();
}

static MlirBytecodeStream
mbci_populateParserPosForSection(MlirBytecodeBytesRef ref) {
  MlirBytecodeStream stream;
  stream.start = stream.pos = ref.data;
  stream.end = stream.start + ref.length;
  return stream;
}

MlirBytecodeStream mlirBytecodeStreamCreate(MlirBytecodeBytesRef bytes) {
  MlirBytecodeStream ret;
  ret.start = ret.pos = bytes.data;
  ret.end = ret.start + bytes.length;
  return ret;
}

static MlirBytecodeBytesRef mbci_getSection(const MlirBytecodeFile *const file,
                                            int index) {
  return file->sectionData[index];
}

MlirBytecodeStatus
mlirBytecodeForEachString(void *callerState,
                          const MlirBytecodeFile *const mlirFile) {
  const MlirBytecodeBytesRef stringSection =
      mbci_getSection(mlirFile, mbci_kString);
  MlirBytecodeStream stream = mbci_populateParserPosForSection(stringSection);

  uint64_t numStrings;
  if (mlirBytecodeFailed(mlirBytecodeParseVarInt(&stream, &numStrings)))
    return mlirBytecodeEmitError("failed to parse number of strings");

  // Parse each of the strings. The sizes of the strings are encoded in reverse
  // order, so that's the order we populate the table.
  size_t stringDataEndOffset = stringSection.length;
  for (uint64_t i = numStrings; i > 0; --i) {
    uint64_t stringSize;
    if (mlirBytecodeFailed(mlirBytecodeParseVarInt(&stream, &stringSize)))
      return mlirBytecodeEmitError("failed to parse string size of string %d",
                                   i - 1);
    if (stringDataEndOffset < stringSize) {
      return mlirBytecodeEmitError(
          "string size exceeds the available data size");
    }

    // Extract the string from the data, dropping the null character.
    size_t stringOffset = stringDataEndOffset - stringSize;
    MlirBytecodeBytesRef str;
    str.data = stringSection.data + stringOffset;
    str.length = stringSize - 1;
    if (mlirBytecodeFailed(mlirBytecodeStringCallBack(
            callerState, (MlirBytecodeStringHandle){.id = i - 1}, numStrings,
            str)))
      return mlirBytecodeEmitError("string callback failed");
    stringDataEndOffset = stringOffset;
  }

  // Check that the only remaining data was for the strings, i.e. the reader
  // should be at the same offset as the first string.
  if (stream.pos != (stream.start + stringDataEndOffset)) {
    return mlirBytecodeEmitError(
        "unexpected trailing data between the offsets for strings "
        "and their data");
  }
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeParseDialectSection(void *callerState,
                                                   MlirBytecodeFile *mlirFile) {
  const MlirBytecodeBytesRef dialectSection =
      mbci_getSection(mlirFile, mbci_kDialect);
  MlirBytecodeStream stream = mbci_populateParserPosForSection(dialectSection);
  mlirBytecodeEmitDebug("parsing dialect section of length %ld",
                        dialectSection.length);

  uint64_t numDialects;
  if (mlirBytecodeFailed(mlirBytecodeParseVarInt(&stream, &numDialects)))
    return mlirBytecodeEmitError("unable to parse number of dialects");
  mlirBytecodeEmitDebug("number of dialects = %d", numDialects);

  uint64_t dialectName;
  for (uint64_t i = 0; i < numDialects; ++i) {
    if (mlirBytecodeFailed(mlirBytecodeParseVarInt(&stream, &dialectName)))
      return mlirBytecodeEmitError("unable to parse dialect %d", i);
    mlirBytecodeEmitDebug("dialect[%d] = %s", i,
                          mlirBytecodeGetStringSectionValue(
                              callerState, mlirFile,
                              (MlirBytecodeDialectHandle){.id = dialectName})
                              .data);
    if (mlirBytecodeFailed(mlirBytecodeDialectCallBack(
            callerState, (MlirBytecodeDialectHandle){.id = i}, numDialects,
            (MlirBytecodeStringHandle){.id = dialectName})))
      return mlirBytecodeFailure();
  }

  MlirBytecodeOpHandle op = {.id = 0};
  while (!mbci_streamEmpty(&stream)) {
    uint64_t dialect, numOpNames;
    mlirBytecodeParseVarInt(&stream, &dialect);
    mlirBytecodeParseVarInt(&stream, &numOpNames);

    mlirBytecodeEmitDebug("parsing for dialect %d %d ops", dialect, numOpNames);
    for (uint64_t j = 0; j < numOpNames; ++j) {
      uint64_t opName;
      if (mlirBytecodeFailed(mlirBytecodeParseVarInt(&stream, &opName)))
        return mlirBytecodeEmitError("failed to parse op name");
      mlirBytecodeEmitDebug(
          "\top[%d] = %s (%d) . %s (%d)", (int)op.id,
          mlirBytecodeGetStringSectionValue(
              callerState, mlirFile, (MlirBytecodeDialectHandle){.id = dialect})
              .data,
          dialect,
          mlirBytecodeGetStringSectionValue(
              callerState, mlirFile, (MlirBytecodeStringHandle){.id = opName})
              .data,
          opName);
      // Associate op[index] = {dialect[j], opName}
      MlirBytecodeStatus ret = mlirBytecodeDialectOpCallBack(
          callerState, op, (MlirBytecodeDialectHandle){.id = dialect},
          (MlirBytecodeStringHandle){.id = opName});
      if (!mlirBytecodeSucceeded(ret))
        return ret;
      ++op.id;
    }
  }

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeForEachAttributeAndType(void *callerState,
                                    MlirBytecodeFile *mlirFile) {
  const MlirBytecodeBytesRef offsetSection =
      mbci_getSection(mlirFile, mbci_kAttrTypeOffset);
  MlirBytecodeStream offsetPP = mbci_populateParserPosForSection(offsetSection);
  const MlirBytecodeBytesRef atSection =
      mbci_getSection(mlirFile, mbci_kAttrType);
  MlirBytecodeStream atPP = mbci_populateParserPosForSection(atSection);

  uint64_t numAttrs, numTypes;
  if (mlirBytecodeFailed(mlirBytecodeParseVarInt(&offsetPP, &numAttrs)) ||
      mlirBytecodeFailed(mlirBytecodeParseVarInt(&offsetPP, &numTypes)))
    return mlirBytecodeEmitError("invalid number of attributes or types");
  mlirBytecodeEmitDebug("parsing %ld attributes and %ld types", numAttrs,
                        numTypes);

  uint64_t i = 0;
  while (i < numAttrs) {
    uint64_t dialect;
    if (mlirBytecodeFailed(mlirBytecodeParseVarInt(&offsetPP, &dialect)))
      return mlirBytecodeEmitError(
          "invalid dialect handle while parsing attributes");

    uint64_t numElements;
    if (mlirBytecodeFailed(mlirBytecodeParseVarInt(&offsetPP, &numElements)))
      return mlirBytecodeEmitError(
          "invalid number of elements in attr offset group");

    for (uint64_t j = 0; j < numElements; ++j) {
      uint64_t length;
      bool hasCustomEncoding;
      if (mlirBytecodeFailed(mlirBytecodeParseVarIntWithFlag(
              &offsetPP, &length, &hasCustomEncoding)))
        return mlirBytecodeEmitError("invalid attr offset");

      MlirBytecodeBytesRef attr = {.data = atPP.pos, .length = length};

      // Verify that the offset is actually valid.
      if (atPP.pos + length > atPP.end) {
        return mlirBytecodeEmitError(
            "attribute or Type entry offset points past the end of section");
      }

      // Parse & associate dialect.attr[j] with `attr`
      MlirBytecodeStatus ret = mlirBytecodeAttrCallBack(
          callerState, (MlirBytecodeAttrHandle){.id = i++}, numAttrs,
          (MlirBytecodeDialectHandle){.id = dialect}, attr, hasCustomEncoding);
      if (mlirBytecodeFailed(ret))
        // TODO: Should we rely that instantiations will always emit error?
        return mlirBytecodeEmitError("attr callback failed");
      if (mlirBytecodeInterrupted(ret))
        break;
      // Unhandled attributes are not considered error.

      if (mlirBytecodeFailed(mbci_skipBytes(&atPP, length)))
        return mlirBytecodeEmitError("invalid attr offset");
    }
  }

  i = 0;
  while (i < numTypes) {
    uint64_t dialect;
    if (mlirBytecodeFailed(mlirBytecodeParseVarInt(&offsetPP, &dialect)))
      return mlirBytecodeEmitError(
          "invalid dialect handle while parsing types");

    uint64_t numElements;
    if (mlirBytecodeFailed(mlirBytecodeParseVarInt(&offsetPP, &numElements)))
      return mlirBytecodeEmitError(
          "invalid number of elements in type offset group");

    for (uint64_t j = 0; j < numElements; ++j) {
      uint64_t offset;
      bool hasCustomEncoding;
      if (mlirBytecodeFailed(mlirBytecodeParseVarIntWithFlag(
              &offsetPP, &offset, &hasCustomEncoding)))
        return mlirBytecodeEmitError("invalid type offset");

      MlirBytecodeBytesRef type = {.data = atPP.pos, .length = offset};
      // Verify that the offset is actually valid.
      if (atPP.pos + offset > atPP.end) {
        return mlirBytecodeEmitError(
            "Attribute or Type entry offset points past the end of section");
      }

      MlirBytecodeStatus ret = mlirBytecodeTypeCallBack(
          callerState, (MlirBytecodeTypeHandle){.id = i++}, numTypes,
          (MlirBytecodeDialectHandle){.id = dialect}, type, hasCustomEncoding);
      if (mlirBytecodeFailed(ret))
        // TODO: Same question as with attributes.
        return mlirBytecodeEmitError("type callback failed");
      if (mlirBytecodeInterrupted(ret))
        break;
      // Unhandled attributes are not considered error.

      if (mlirBytecodeFailed(mbci_skipBytes(&atPP, offset)))
        return mlirBytecodeEmitError("invalid type offset");
    }
  }

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeForEachResource(void *callerState,
                            MlirBytecodeFile *const mlirFile) {
  const MlirBytecodeBytesRef offsetSection =
      mbci_getSection(mlirFile, mbci_kResourceOffset);
  MlirBytecodeStream offsetPP = mbci_populateParserPosForSection(offsetSection);
  const MlirBytecodeBytesRef resSection =
      mbci_getSection(mlirFile, mbci_kResourceOffset);
  MlirBytecodeStream resPP = mbci_populateParserPosForSection(resSection);

  uint64_t numExternalResourceGroups;
  if (mlirBytecodeFailed(
          mlirBytecodeParseVarInt(&offsetPP, &numExternalResourceGroups)))
    return mlirBytecodeEmitError(
        "invalid/missing number of external resource groups");
  if (!numExternalResourceGroups)
    return mlirBytecodeSuccess();

  for (uint64_t i = 0; i < numExternalResourceGroups; ++i) {
    uint64_t groupKey;
    if (mlirBytecodeFailed(mlirBytecodeParseVarInt(&offsetPP, &groupKey)))
      return mlirBytecodeEmitError("invalid/missing resource group key");
    mlirBytecodeEmitDebug(
        "Group for %s / %d",
        mlirBytecodeGetStringSectionValue(
            callerState, mlirFile, (MlirBytecodeStringHandle){.id = groupKey})
            .data,
        numExternalResourceGroups);

    uint64_t numResources;
    if (mlirBytecodeFailed(mlirBytecodeParseVarInt(&offsetPP, &numResources)))
      return mlirBytecodeEmitError("invalid/missing number of resources");

    for (uint64_t j = 0; j < numResources; ++j) {
      uint64_t resourceKey;
      if (mlirBytecodeFailed(mlirBytecodeParseVarInt(&offsetPP, &resourceKey)))
        return mlirBytecodeEmitError("invalid/missing resource key");
      uint64_t size;
      if (mlirBytecodeFailed(mlirBytecodeParseVarInt(&offsetPP, &size)))
        return mlirBytecodeEmitError("invalid/missing resource size");
      uint8_t kind;
      if (mlirBytecodeFailed(mbci_parseByte(&offsetPP, &kind)))
        return mlirBytecodeEmitError("invalid/missing resource kind");

      MlirBytecodeAsmResourceEntryKind resKind = (MlirBytecodeAsmResourceEntryKind)kind;
      switch (resKind) {
      case kMlirBytecodeResourceEntryBool: {
        uint8_t value;
        if (mlirBytecodeFailed(mbci_parseByte(&resPP, &value)))
          return mlirBytecodeEmitError("invalid/missing resource entry");
        if (mlirBytecodeFailed(mlirBytecodeResourceCallBack(
                callerState, (MlirBytecodeStringHandle){.id = groupKey},
                numExternalResourceGroups,
                (MlirBytecodeStringHandle){.id = resourceKey}, resKind, NULL,
                &value, NULL)))
          return mlirBytecodeFailure();
        break;
      }
      case kMlirBytecodeResourceEntryBlob: {
        uint64_t alignment;
        if (mlirBytecodeFailed(mlirBytecodeParseVarInt(&resPP, &alignment)))
          return mlirBytecodeEmitError("invalid/missing resource alignment");
        uint64_t size;
        if (mlirBytecodeFailed(mlirBytecodeParseVarInt(&resPP, &size)))
          return mlirBytecodeEmitError("invalid/missing resource size");
        if (mlirBytecodeFailed(mbci_alignTo(&resPP, alignment)))
          return mlirBytecodeFailure();
        MlirBytecodeBytesRef blob = {.data = resPP.pos, .length = size};
        if (mlirBytecodeFailed(mlirBytecodeResourceCallBack(
                callerState, (MlirBytecodeStringHandle){.id = groupKey},
                numExternalResourceGroups,
                (MlirBytecodeStringHandle){.id = resourceKey}, resKind, &blob,
                NULL, NULL)))
          return mlirBytecodeFailure();
        break;
      }
      case kMlirBytecodeResourceEntryString: {
        uint64_t value;
        if (mlirBytecodeFailed(mlirBytecodeParseVarInt(&resPP, &value)))
          return mlirBytecodeEmitError("invalid/missing resource value");
        if (mlirBytecodeFailed(mlirBytecodeResourceCallBack(
                callerState, (MlirBytecodeStringHandle){.id = groupKey},
                numExternalResourceGroups,
                (MlirBytecodeStringHandle){.id = resourceKey}, resKind, NULL,
                NULL, (MlirBytecodeStringHandle *)&value)))
          return mlirBytecodeFailure();
        break;
      }
      }
    }
  }
  return mlirBytecodeSuccess();
}

enum {
  // Sentinel value to indicate unset number of blocks & ops remaining.
  mbci_MlirIRSectionStackUnset = -1,
};

// We keep the following on the stack for a op with regions when descending into
// regions:
struct mbci_MlirIRSectionStackEntry {
  // State of parent op as currently populated.
  MlirBytecodeOperationStateHandle op;

  // Number of regions to still complete parsing.
  // Number of regions is assumed to be relatively small (<= 32767).
  uint16_t numRegionsRemaining : 15;

  // Whether the operation is isolated from above.
  bool isIsolatedFromAbove : 1;

  // Number of blocks to still complete parsing.
  // Number of regions is assumed to be relatively small (<= 65536).
  uint16_t numBlocksRemaining;

  // Number of ops remaining.
  uint32_t numOpsRemaining;
};
typedef struct mbci_MlirIRSectionStackEntry mbci_MlirIRSectionStackEntry;

// IR section parsing stack. Instruction without region does not result in
// pushing anything on stack, for operations with regions:
struct mbci_MlirIRSectionStack {
  int top;

  mbci_MlirIRSectionStackEntry data[MlirIRSectionStackMaxDepth + 1];
};
typedef struct mbci_MlirIRSectionStack mbci_MlirIRSectionStack;

static mbci_MlirIRSectionStackEntry *
mbci_getStackTop(mbci_MlirIRSectionStack *stack) {
  return &stack->data[stack->top];
}
static mbci_MlirIRSectionStackEntry *
mbci_getStackWIP(mbci_MlirIRSectionStack *stack) {
  return &stack->data[stack->top + 1];
}

// Initialize stack entry with required state to resume. `op` is initialized
// already and so not passed in.
static MlirBytecodeStatus
mbci_mlirIRSectionStackPush(mbci_MlirIRSectionStack *stack) {
  if (stack->top == MlirIRSectionStackMaxDepth) {
    mlirBytecodeEmitError("IR stack max depth exceeded");
    return mlirBytecodeIterationInterrupt();
  }
  ++stack->top;
  mbci_MlirIRSectionStackEntry *it = mbci_getStackTop(stack);
  it->numBlocksRemaining = mbci_MlirIRSectionStackUnset;
  it->numOpsRemaining = mbci_MlirIRSectionStackUnset;

  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus
mbci_mlirIRSectionStackPop(mbci_MlirIRSectionStack *stack) {
  --stack->top;
  return mlirBytecodeSuccess();
}

#ifdef NDEBUG
#define MLIRBC_AFTER(EXP, OBS, NUM)
#else
#define MLIRBC_AFTER(EXP, OBS, NUM)                                            \
  do {                                                                         \
    assert(mlirBytecodeSucceeded(mlirBytecodeSkipHandles(&EXP, NUM)));         \
    if (EXP.pos != OBS->pos) {                                                 \
      mlirBytecodeEmitDebug("mismatch: %" PRIu64 " vs %" PRIu64 "\n", EXP.pos, \
                            OBS->pos);                                         \
    }                                                                          \
    assert(EXP.pos == OBS->pos);                                               \
  } while (false);
#endif

static MlirBytecodeStatus mbci_parseBlock(MlirBytecodeStream *stream,
                                          mbci_MlirIRSectionStack *stack,
                                          void *callerState) {
  mbci_MlirIRSectionStackEntry *top = mbci_getStackTop(stack);
  bool hasArgs;
  uint64_t numOpsRemaining;
  if (mlirBytecodeFailed(
          mlirBytecodeParseVarIntWithFlag(stream, &numOpsRemaining, &hasArgs)))
    return mlirBytecodeEmitError("unable to parse block's number of args");
  top->numOpsRemaining = numOpsRemaining;
  mlirBytecodeEmitDebug("num ops = %d and block %s", numOpsRemaining,
                        hasArgs ? "has args" : "has no args");

  uint64_t numArgs = 0;
#ifndef NDEBUG
  MlirBytecodeStream expectedEnd = *stream;
#endif
  if (hasArgs) {
    if (mlirBytecodeFailed(mlirBytecodeParseVarInt(stream, &numArgs)))
      return mlirBytecodeEmitError("invalid/missing number of block args");
  }

  MlirBytecodeStatus ret = mlirBytecodeOperationStateBlockPush(
      callerState, top->op, stream, numArgs);
  MLIRBC_AFTER(expectedEnd, stream, 2 * numArgs + hasArgs);
  return ret;
}

static MlirBytecodeStatus mbci_parseRegion(MlirBytecodeStream *stream,
                                           mbci_MlirIRSectionStack *stack,
                                           void *state) {
  mbci_MlirIRSectionStackEntry *top = mbci_getStackTop(stack);
  uint64_t numBlocksRemaining;
  if (mlirBytecodeFailed(mlirBytecodeParseVarInt(stream, &numBlocksRemaining)))
    return mlirBytecodeEmitError("invalid number of blocks");
  top->numBlocksRemaining = numBlocksRemaining;
  top->numOpsRemaining = mbci_MlirIRSectionStackUnset;
  if (top->numBlocksRemaining == 0)
    return mlirBytecodeSuccess();

  uint64_t numValues;
  if (mlirBytecodeFailed(mlirBytecodeParseVarInt(stream, &numValues)))
    return mlirBytecodeEmitError("invalid/missing number of values in region");
  mlirBytecodeEmitDebug("numBlocks = %d numValues = %d", numBlocksRemaining,
                        numValues);

  return mlirBytecodeRegionPush(state, top->op, top->numBlocksRemaining,
                                numValues);
}

static MlirBytecodeStatus mbci_parseOperation(MlirBytecodeStream *stream,
                                              mbci_MlirIRSectionStack *stack,
                                              void *callerState) {
  enum OpEncodingMask {
    kHasAttrs = 0x01,
    kHasResults = 0x02,
    kHasOperands = 0x04,
    kHasSuccessors = 0x08,
    kHasInlineRegions = 0x10,
  };

  MlirBytecodeOpHandle name;
  if (mlirBytecodeFailed(mlirBytecodeParseVarInt(stream, &name.id)))
    return mlirBytecodeEmitError("invalid operation");

  uint8_t encodingMask;
  if (mlirBytecodeFailed(mbci_parseByte(stream, &encodingMask)))
    return mlirBytecodeEmitError("invalid encoding mask");

  MlirBytecodeLocHandle loc;
  if (mlirBytecodeFailed(mlirBytecodeParseVarInt(stream, &loc.id)))
    return mlirBytecodeEmitError("invalid operation location");

  mbci_MlirIRSectionStackEntry *cur = mbci_getStackWIP(stack);
  MlirBytecodeStatus ret =
      mlirBytecodeOperationStatePush(callerState, name, loc, &cur->op);
  if (!mlirBytecodeSucceeded(ret))
    return ret;

  if (encodingMask & kHasAttrs) {
    MlirBytecodeAttrHandle attrDict;
    if (mlirBytecodeFailed(mlirBytecodeParseHandle(stream, &attrDict))) {
      return mlirBytecodeEmitError("invalid op attribute handle");
    }
    ret =
        mlirBytecodeOperationStateAddAttributes(callerState, cur->op, attrDict);
    if (!mlirBytecodeSucceeded(ret))
      return mlirBytecodeEmitDebug("unable to add attributes"), ret;
  }

  // Parsing all the variadic sizes.
  if (encodingMask & kHasResults) {
    uint64_t numResults = 0;
    if (mlirBytecodeSucceeded(mlirBytecodeParseVarInt(stream, &numResults))) {
#ifndef NDEBUG
      MlirBytecodeStream expectedEnd = *stream;
#endif
      ret = mlirBytecodeOperationStateAddResultTypes(callerState, cur->op,
                                                     stream, numResults);
      if (!mlirBytecodeSucceeded(ret))
        return mlirBytecodeEmitError("invalid result type"), ret;
      MLIRBC_AFTER(expectedEnd, stream, numResults);
    } else {
      return mlirBytecodeEmitError("invalid number of results");
    }
  }

  if (encodingMask & kHasOperands) {
    uint64_t numOperands = 0;
    if (mlirBytecodeSucceeded(mlirBytecodeParseVarInt(stream, &numOperands))) {
#ifndef NDEBUG
      MlirBytecodeStream expectedEnd = *stream;
#endif
      ret = mlirBytecodeOperationStateAddOperands(callerState, cur->op, stream,
                                                  numOperands);
      if (!mlirBytecodeSucceeded(ret))
        return mlirBytecodeEmitError("invalid operands"), ret;

      MLIRBC_AFTER(expectedEnd, stream, numOperands);
    } else {
      return mlirBytecodeEmitError("invalid number of operands");
    }
  }

  if (encodingMask & kHasSuccessors) {
    uint64_t numSuccessors = 0;
    if (mlirBytecodeSucceeded(
            mlirBytecodeParseVarInt(stream, &numSuccessors))) {
#ifndef NDEBUG
      MlirBytecodeStream expectedEnd = *stream;
#endif

      ret = mlirBytecodeOperationStateAddSuccessors(callerState, cur->op,
                                                    stream, numSuccessors);
      if (!mlirBytecodeSucceeded(ret))
        return mlirBytecodeEmitError("invalid successors"), ret;

      MLIRBC_AFTER(expectedEnd, stream, numSuccessors);
    } else {
      return mlirBytecodeEmitError("invalid number of successors");
    }
  }

  if (encodingMask & kHasInlineRegions) {
    uint64_t numRegions = 0;
    bool isIsolatedFromAbove = false;
    if (mlirBytecodeFailed(mlirBytecodeParseVarIntWithFlag(
            stream, &numRegions, &isIsolatedFromAbove))) {
      return mlirBytecodeEmitError("invalid number of regions");
    }
    cur->numRegionsRemaining = numRegions;
    cur->isIsolatedFromAbove = isIsolatedFromAbove;
    ret =
        mlirBytecodeOperationStateAddRegions(callerState, cur->op, numRegions);
    if (!mlirBytecodeSucceeded(ret))
      return ret;
    return mbci_mlirIRSectionStackPush(stack);
  }

  // Completed parsing of op.
  ret = mlirBytecodeOperationStatePop(callerState, cur->op);
  if (!mlirBytecodeSucceeded(ret))
    return ret;
  // `cur` is the scratchpad for parsing op without region while `top` is where
  // it is being parsed.
  --mbci_getStackTop(stack)->numOpsRemaining;
  return ret;
}

MlirBytecodeStatus mlirBytecodeParseIRSection(void *callerState,
                                              MlirBytecodeFile *mlirFile) {
  const MlirBytecodeBytesRef irSection = mbci_getSection(mlirFile, mbci_kIR);
  MlirBytecodeStream stream = mbci_populateParserPosForSection(irSection);

  bool hasArgs;
  uint64_t numOps;
  if (mlirBytecodeFailed(
          mlirBytecodeParseVarIntWithFlag(&stream, &numOps, &hasArgs)))
    return mlirBytecodeFailure();
  if (hasArgs)
    return mlirBytecodeEmitError("IR section may not have block args");
  if (numOps != 1)
    return mlirBytecodeEmitError("only one top-level op supported");

  // Initialize with parsing of top-level.
  mbci_MlirIRSectionStack stack = {.top = -1};
  mbci_parseOperation(&stream, &stack, callerState);

  while (stack.top != -1) {
    mbci_MlirIRSectionStackEntry *top = mbci_getStackTop(&stack);

    MlirBytecodeStatus ret;
    bool next = false;
    if (top->numOpsRemaining == 0) {
      mlirBytecodeEmitDebug("no ops remaining in block");
      ret = mlirBytecodeBlockPop(callerState, top->op);
      if (!mlirBytecodeSucceeded(ret))
        return ret;

      --top->numBlocksRemaining;
      top->numOpsRemaining = mbci_MlirIRSectionStackUnset;
      next = true;
    }
    if (top->numBlocksRemaining == 0) {
      mlirBytecodeEmitDebug("no blocks remaining in region");
      ret = mlirBytecodeRegionPop(callerState, top->op);
      if (!mlirBytecodeSucceeded(ret))
        return ret;

      --top->numRegionsRemaining;
      top->numBlocksRemaining = mbci_MlirIRSectionStackUnset;
      next = true;
    }
    if (top->numRegionsRemaining == 0) {
      mlirBytecodeEmitDebug("no regions remaining in op");
      ret = mlirBytecodeOperationStatePop(callerState, top->op);
      if (!mlirBytecodeSucceeded(ret))
        return ret;

      mbci_mlirIRSectionStackPop(&stack);
      --mbci_getStackTop(&stack)->numOpsRemaining;
      next = true;
    }
    if (next)
      continue;

    if (top->numBlocksRemaining == (uint16_t)mbci_MlirIRSectionStackUnset) {
      mlirBytecodeEmitDebug("starting region parsing");
      ret = mbci_parseRegion(&stream, &stack, callerState);
      if (!mlirBytecodeSucceeded(ret))
        return ret;
      continue;
    }
    if (top->numOpsRemaining == (uint32_t)mbci_MlirIRSectionStackUnset) {
      mlirBytecodeEmitDebug("starting block parsing");
      ret = mbci_parseBlock(&stream, &stack, callerState);
      if (!mlirBytecodeSucceeded(ret))
        return ret;
      continue;
    }
    mlirBytecodeEmitDebug(
        "parsing operation (remaining regions: %d blocks: %d ops: %d)",
        (int)mbci_getStackTop(&stack)->numRegionsRemaining,
        (int)mbci_getStackTop(&stack)->numBlocksRemaining,
        (int)mbci_getStackTop(&stack)->numOpsRemaining);
    ret = mbci_parseOperation(&stream, &stack, callerState);
    if (!mlirBytecodeSucceeded(ret))
      return ret;
  }
  return mlirBytecodeSuccess();
}

bool mlirBytecodeFileEmpty(MlirBytecodeFile *file) {
  return file->producer.length == 0;
}

MlirBytecodeFile mlirBytecodePopulateFile(const MlirBytecodeBytesRef bytes) {
  MlirBytecodeFile ret;
  memset(&ret, 0, sizeof(MlirBytecodeFile));
  MlirBytecodeStream stream = mbci_populateParserPosForSection(bytes);

  uint8_t magic[4];
  if (mlirBytecodeFailed(mbci_parseBytes(&stream, magic, 4)))
    return mlirBytecodeEmitError("unable to read 4 byte magic code"), ret;
  if (magic[0] != 'M' || magic[1] != 'L' || magic[2] != 0xef || magic[3] != 'R')
    return mlirBytecodeEmitError("invalid file magic code"), ret;

  // Parse the bytecode version and producer.
  MlirBytecodeBytesRef producer;
  if (mlirBytecodeFailed(mlirBytecodeParseVarInt(&stream, &ret.version)) ||
      mlirBytecodeFailed(
          mlirBytecodeParseNullTerminatedString(&stream, &producer)))
    return mlirBytecodeEmitError("invalid version or producer"), ret;
  mlirBytecodeEmitDebug("bytecode producer: %s\n", producer.data);

  while (!mbci_streamEmpty(&stream)) {
    // Read the next section from the bytecode.
    if (mlirBytecodeFailed(mbci_parseSections(&stream, ret.sectionData)))
      return mlirBytecodeEmitError("invalid section"), ret;
  }

  // Check that all of the required sections were found
  for (int i = 0; i < mbci_kNumSections; ++i) {
    if (mbci_getSection(&ret, i).data == NULL && !mbci_isSectionOptional(i)) {
      return mlirBytecodeEmitError("missing data for top-level section: ",
                                   mbci_sectionIDToString(i)),
             ret;
    }
  }
  // Mark as valid.
  ret.producer = producer;
  return ret;
}

MlirBytecodeStatus mlirBytecodeParseFile(void *callerState,
                                         MlirBytecodeBytesRef bytes) {
  MlirBytecodeFile mlirFile = mlirBytecodePopulateFile(bytes);
  if (mlirBytecodeFileEmpty(&mlirFile))
    return mlirBytecodeFailure();

  // Process the string section first.
  if (mlirBytecodeFailed(mlirBytecodeForEachString(callerState, &mlirFile)) ||
      // Process the dialect section.
      mlirBytecodeFailed(
          mlirBytecodeParseDialectSection(callerState, &mlirFile)) ||
      // Process the resource section if present.
      mlirBytecodeFailed(mlirBytecodeForEachResource(callerState, &mlirFile)) ||
      // Process the attribute and type section.
      mlirBytecodeFailed(
          mlirBytecodeForEachAttributeAndType(callerState, &mlirFile)) ||
      // Finally, process the IR section.
      mlirBytecodeFailed(mlirBytecodeParseIRSection(callerState, &mlirFile)))
    return mlirBytecodeFailure();

  return mlirBytecodeSuccess();
}
