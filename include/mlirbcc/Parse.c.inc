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
// entry point is `mlirBytecodeParse`.
//
// Callbacks/functions to be implemented in instantiation accept an opaque
// pointer as first argument that gets directly propagated during parsing and
// can be used by instantiation for additional/parse local state capture.
//
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
#define MLIRBC_IR_STACK_MAX_DEPTH 15 // TODO: add dynamic option.

typedef struct MlirBytecodeStatus MlirBytecodeStatus;

//===----------------------------------------------------------------------===//
// MLIR file parsing methods.

/// Populates the MlirBytecodeParserState contents for given in memory bytes.
/// Returns an empty file if population failed.
static MlirBytecodeParserState
mlirBytecodePopulateParserState(void *callerState, MlirBytecodeBytesRef bytes);

/// Parses the given MLIR file represented in memory `bytes`, calls the
/// appropriate callbacks during parsing. This combines the parsing methods
/// below.
static MlirBytecodeStatus mlirBytecodeParse(void *callerState,
                                            MlirBytecodeBytesRef bytes);

/// Returns whether the given MlirBytecodeParserState structure is empty.
static bool mlirBytecodeParserStateEmpty(MlirBytecodeParserState *file);

//===----------------------------------------------------------------------===//
// Functions required to be defined:
#define MLIRBC_DEF static

/// Hook to populate handle to a operation state (e.g., helper structure to
/// collect fields required to define operation).
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStatePush(
    void *callerState, MlirBytecodeOpHandle name, MlirBytecodeLocHandle loc,
    MlirBytecodeOperationStateHandle *opState);

/// Called when finalizing population of operation state.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStatePop(
    void *callerState, MlirBytecodeOperationStateHandle);

/// Called for an operation with attributes.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStateAddAttributeDictionary(
    void *callerState, MlirBytecodeOperationStateHandle,
    MlirBytecodeAttrHandle);

/// Called for operation with results.
// Note: This function returns reference to unpacked array which may be
// overwritten by following parse calls, hence either copy or process.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStateAddResultTypes(
    void *callerState, MlirBytecodeOperationStateHandle,
    MlirBytecodeHandlesRef);

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
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStateBlockPop(
    void *callerState, MlirBytecodeOperationStateHandle);

/// Called when exiting a region.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStateRegionPop(
    void *callerState, MlirBytecodeOperationStateHandle);

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

/// Invoked with dialect and attribute with the attribute's range in memory.
// TODO: This and others currently pass in total number of attributes per
// invocation, consider factoring out to an initialize that sets total size.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeAttrCallBack(
    void *, MlirBytecodeAttrHandle, size_t total, MlirBytecodeDialectHandle,
    MlirBytecodeBytesRef, bool hasCustom);

/// Invoked with dialect and type with the type's range in memory.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeTypeCallBack(
    void *, MlirBytecodeTypeHandle, size_t total, MlirBytecodeDialectHandle,
    MlirBytecodeBytesRef, bool hasCustom);

/// Invoked with dialect handle and its corresponding string handle.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeDialectCallBack(void *, MlirBytecodeDialectHandle, size_t /*total*/,
                            MlirBytecodeStringHandle);

/// Invoked with opname with its dialect and string handle corresponding to its
/// name.
static MlirBytecodeStatus
mlirBytecodeDialectOpCallBack(void *, MlirBytecodeDialectHandle,
                              MlirBytecodeOpHandle, MlirBytecodeStringHandle);

/// Invoked when starting parsing of resources.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeResourceSectionEnter(
    void *, MlirBytecodeSize numExternalResourceGroups);

MLIRBC_DEF MlirBytecodeStatus mlirBytecodeResourceGroupEnter(
    void *, MlirBytecodeStringHandle groupKey, MlirBytecodeSize numResources);

/// Invoked per blob resource in group.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeResourceBlobCallBack(
    void *, MlirBytecodeStringHandle groupKey,
    MlirBytecodeStringHandle resourceKey, MlirBytecodeBytesRef blob);
/// Invoked per bool resource in group.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeResourceBoolCallBack(
    void *, MlirBytecodeStringHandle groupKey,
    MlirBytecodeStringHandle resourceKey, const uint8_t);
/// Invoked per string resource in group.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeResourceStringCallBack(
    void *, MlirBytecodeStringHandle groupKey,
    MlirBytecodeStringHandle resourceKey, MlirBytecodeStringHandle);

/// Invoked with string handle, total number of strings in string section and
/// bytes corresponding to the string per entry in string section.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeStringCallBack(
    void *, MlirBytecodeStringHandle, size_t total, MlirBytecodeBytesRef);

// Returns the requested string from the string section.
// It is only used for debugging messages in Parse.h.
MLIRBC_DEF MlirBytecodeBytesRef mlirBytecodeGetStringSectionValue(
    void *callerState, MlirBytecodeStringHandle idx);

//===----------------------------------------------------------------------===//
// Section parsing entry points.

// Parse and associate attr handle with parsed attribute.
typedef MlirBytecodeStatus (*MlirBytecodeAttrParseCallBack)(
    void *, MlirBytecodeAttrHandle, MlirBytecodeBytesRef, bool hasCustom);

// Parse and associate type handle with parsed type.
typedef MlirBytecodeStatus (*MlirBytecodeTypeParseCallBack)(
    void *, MlirBytecodeTypeHandle, MlirBytecodeBytesRef, bool hasCustom);

/// Iterators over attributes and types, calling MlirBytecodeAttrCallBack and
/// MlirBytecodeTypeCallBack upon encountering Attribute or Type respectively.
/// Returns whether failed.
static MlirBytecodeStatus
mlirBytecodeParseAttributeAndTypeSection(void *callerState,
                                         MlirBytecodeParserState *parserState);

/// Parses the dialect section, invoking MlirBytecodeDialectCallBack upon
/// dialect encountered and MlirBytecodeDialectOpCallBack per operation type in
/// dialect. Returns whether failed.
static MlirBytecodeStatus
mlirBytecodeParseDialectSection(void *callerState,
                                MlirBytecodeParserState *parserState);

/// Invoked with opname with its dialect and string handle corresponding to its
/// name.
static MlirBytecodeStatus
mlirBytecodeDialectOpCallBack(void *callerState, MlirBytecodeDialectHandle,
                              MlirBytecodeOpHandle, MlirBytecodeStringHandle);

/// Parse IR section. The block args, operation and region callback
/// are invoked during bytecode in-order walk. Additionally allows for passing
/// in an opaque state.
///
/// The IR section parsing follows the nesting order:
///   op ->* regions ->* blocks
/// The caller is required to keep track of when all operations/blocks in
/// block/region have been processed and so parsing resumes at parent level.
/// Returns whether failed.
static MlirBytecodeStatus
mlirBytecodeParseIRSection(void *callerState,
                           MlirBytecodeParserState *parserState);

/// Parse the resource section, calling MlirBytecodeResourceCallBack upon
/// resources encountered. Returns whether failed.
static MlirBytecodeStatus
mlirBytecodeParseResourceSection(void *callerState,
                                 MlirBytecodeParserState *parserState);

//===----------------------------------------------------------------------===//
// Dialect parsing utility methods.

/// Parse the next handle on the stream and increment stream.
static MlirBytecodeStatus mlirBytecodeParseHandle(void *callerState,
                                                  MlirBytecodeStream *stream,
                                                  MlirBytecodeHandle *result);

/// Skips the next n handles.
static MlirBytecodeStatus mlirBytecodeSkipHandles(void *callerState,
                                                  MlirBytecodeStream *stream,
                                                  uint64_t n);

/// Parse byte from stream.
static MlirBytecodeStatus mlirBytecodeParseByte(void *callerState,
                                                MlirBytecodeStream *stream,
                                                uint8_t *val);

/// Parse n bytes from stream.
static MlirBytecodeStatus mlirBytecodeParseBytes(void *callerState,
                                                 MlirBytecodeStream *stream,
                                                 size_t n, const uint8_t **val);

/// Parse uint64 on the stream and increment stream.
static MlirBytecodeStatus mlirBytecodeParseVarInt(void *callerState,
                                                  MlirBytecodeStream *stream,
                                                  uint64_t *result);

/// Parse int64 on the stream and increment stream.
static MlirBytecodeStatus
mlirBytecodeParseSignedVarInt(void *callerState, MlirBytecodeStream *stream,
                              int64_t *result);

/// Parse a variable length encoded integer whose low bit is used to encode a
/// flag, i.e: `(integerValue << 1) | (flag ? 1 : 0)`.
static MlirBytecodeStatus
mlirBytecodeParseVarIntWithFlag(void *callerState, MlirBytecodeStream *stream,
                                uint64_t *result, bool *flag);

/// Parse a null-terminated string.
static MlirBytecodeStatus mlirBytecodeParseNullTerminatedString(
    void *callerState, MlirBytecodeStream *stream, MlirBytecodeBytesRef *str);

//===----------------------------------------------------------------------===//
// MlirBytecodeStatus.

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

/// Helper function to emit error and return failure status.
extern MlirBytecodeStatus mlirBytecodeEmitErrorImpl(void *callerState,
                                                    const char *fmt, ...);

/// Helper function to emit debugging info.
extern void mlirBytecodeEmitDebugImpl(const char *file, int line,
                                      const char *fmt, ...);

#ifdef __cplusplus
}
#endif

//===----------------------------------------------------------------------===//
// Implementation/implementation details.

// Macros to allow capturing file and line from which invoked.
#define mlirBytecodeEmitDebug(...)                                             \
  mlirBytecodeEmitDebugImpl(__FILE__, __LINE__, __VA_ARGS__)

#ifdef MLIRBC_VERBOSE_ERROR
#define mlirBytecodeEmitError(callerState, ...)                                \
  mlirBytecodeEmitErrorImpl(callerState, __VA_ARGS__)
#else
#define mlirBytecodeEmitError(callerState, ...) mlirBytecodeFailure()
#endif

#if __has_builtin(__builtin_expect) || defined(__GNUC__)
#define MLIRBC_LIKELY(EXPR) __builtin_expect((bool)(EXPR), true)
#define MLIRBC_UNLIKELY(EXPR) __builtin_expect((bool)(EXPR), false)
#else
#define MLIRBC_LIKELY(EXPR) (EXPR)
#define MLIRBC_UNLIKELY(EXPR) (EXPR)
#endif

//===----------------------------------------------------------------------===//
// Bytecode constants.

enum {
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

static const char *mbci_sectionIDToString(uint8_t id) {
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
// Parser state definition and helpers.

struct MlirBytecodeParserState {
  uint64_t version;
  MlirBytecodeBytesRef producer;

  MlirBytecodeBytesRef sectionData[mbci_kNumSections];

  // Scratch pad for unpacking handles.
  MlirBytecodeHandle *scratchBase;
  size_t size, capacity;
};

MlirBytecodeStatus mbci_pushOnScratch(MlirBytecodeParserState* state, size_t n, MlirBytecodeHandle** res) {
  if (MLIRBC_UNLIKELY(state->size + n > state->capacity)) {
    // Something smarter could be done here, but expected common case is
    // reasonable initial capacity set by user and most scratch space usage is
    // smaller.
    state->capacity += n;
    state->scratchBase = (MlirBytecodeHandle*)realloc(state->scratchBase, state->capacity * sizeof(MlirBytecodeHandle));
  }

  *res = state->scratchBase + state->size;
  state->size += n;
  return mlirBytecodeSuccess();
}

void mbci_clearScratch(MlirBytecodeParserState* state, size_t n) {
  state->size = 0;
}

//===----------------------------------------------------------------------===//
// Default implementations for verbose logging and debugging
#ifdef MLIRBC_VERBOSE_ERROR
__attribute__((weak)) MlirBytecodeStatus
mlirBytecodeEmitErrorImpl(void *callerState, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");

  return mlirBytecodeFailure();
}
#else
MlirBytecodeStatus mlirBytecodeEmitErrorImpl(void *callerState, const char *fmt,
                                             ...) {
  (void)fmt;
  return mlirBytecodeFailure();
}
#endif

#ifdef MLIRBC_DEBUG
__attribute__((weak)) void mlirBytecodeEmitDebugImpl(const char *file, int line,
                                                     const char *fmt, ...) {
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

static MlirBytecodeStatus mlirBytecodeParseByte(void *callerState,
                                                MlirBytecodeStream *stream,
                                                uint8_t *val) {
  if (MLIRBC_UNLIKELY(mbci_streamEmpty(stream)))
    return mlirBytecodeFailure();
  *val = *stream->pos++;
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus mlirBytecodeParseBytes(void *callerState,
                                                 MlirBytecodeStream *stream,
                                                 size_t n,
                                                 const uint8_t **result) {
  *result = stream->pos;
  stream->pos += n;
  if (MLIRBC_UNLIKELY(stream->pos > stream->end))
    return mlirBytecodeFailure();
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus mbci_skipBytes(void *callerState,
                                         MlirBytecodeStream *stream, size_t n) {
  stream->pos += n;
  if (MLIRBC_UNLIKELY(stream->pos > stream->end))
    return mlirBytecodeFailure();
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus
mbci_skipVarInts(void *callerState, MlirBytecodeStream *stream, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    uint8_t head;
    MlirBytecodeStatus ret = mlirBytecodeParseByte(callerState, stream, &head);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeFailure();
    int numBytes = (MLIRBC_UNLIKELY(head == 0)) ? 8 : __builtin_ctz(head);
    ret = mbci_skipBytes(callerState, stream, numBytes);
    if (mlirBytecodeFailed(ret))
      return mlirBytecodeFailure();
  }
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeParseHandle(void *callerState,
                                           MlirBytecodeStream *stream,
                                           MlirBytecodeAttrHandle *result) {
  return mlirBytecodeParseVarInt(callerState, stream, &result->id);
}

__attribute__((used)) MlirBytecodeStatus mlirBytecodeSkipHandles(void *callerState,
                                           MlirBytecodeStream *stream,
                                           uint64_t n) {
  return mbci_skipVarInts(callerState, stream, n);
}

MlirBytecodeStatus mlirBytecodeParseVarInt(void *callerState,
                                           MlirBytecodeStream *stream,
                                           uint64_t *result) {
  // Compute the number of bytes needed to encode the value. Each byte can hold
  // up to 7-bits of data. We only check up to the number of bits we can encode
  // in the first byte (8).
  uint8_t head;
  if (mlirBytecodeFailed(mlirBytecodeParseByte(callerState, stream, &head)))
    return mlirBytecodeFailure();
  *result = head;
  if ((*result & 1)) {
    *result >>= 1;
    return mlirBytecodeSuccess();
  }

  uint32_t numBytes = (MLIRBC_UNLIKELY(head == 0)) ? 8 : __builtin_ctz(head);
  const uint8_t *res;
  MlirBytecodeStatus ret =
      mlirBytecodeParseBytes(callerState, stream, numBytes, &res);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeFailure();
  uint8_t *ptr = (uint8_t *)(result) + 1;
  for (uint32_t i = 0; i < numBytes; ++i)
    ptr[i] = res[i];

  // Shift out the low-order bits that were used to mark how the value was
  // encoded.
  *result >>= (numBytes + 1);
  return ret;
}

MlirBytecodeStatus mlirBytecodeParseVarIntWithFlag(void *callerState,
                                                   MlirBytecodeStream *stream,
                                                   uint64_t *result,
                                                   bool *flag) {
  MlirBytecodeStatus ret = mlirBytecodeParseVarInt(callerState, stream, result);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeFailure();
  *flag = *result & 1;
  *result >>= 1;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeParseSignedVarInt(void *callerState,
                                                 MlirBytecodeStream *stream,
                                                 int64_t *result) {
  MlirBytecodeStatus ret =
      mlirBytecodeParseVarInt(callerState, stream, (uint64_t *)result);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeFailure();
  *result = (*result >> 1) ^ (~(*result & 1) + 1);
  return mlirBytecodeSuccess();
}

// Align the current reader position to the specified alignment.
static MlirBytecodeStatus mbci_alignTo(void *callerState,
                                       MlirBytecodeStream *stream,
                                       uint32_t alignment) {
  bool isPowerOf2 = (alignment != 0) && ((alignment & (alignment - 1)) == 0);
  if (MLIRBC_UNLIKELY(!isPowerOf2))
    return mlirBytecodeEmitError(callerState,
                                 "expected alignment to be a power-of-two");

  // An arbitrary value used to fill alignment padding.
  const uint8_t kAlignmentByte = 0xCB;

  // Shift the reader position to the next alignment boundary.
  while ((uintptr_t)stream->pos & ((uintptr_t)alignment - 1)) {
    uint8_t padding;
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(
            mlirBytecodeParseByte(callerState, stream, &padding))))
      return mlirBytecodeFailure();
    if (padding != kAlignmentByte) {
      return mlirBytecodeEmitError(
          callerState, "expected alignment byte (0x%x), but got: '0x%x'",
          kAlignmentByte, padding);
    }
  }

  // TODO: Check that the current data pointer is actually at the expected
  // alignment.
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus
mbci_parseSections(void *callerState, MlirBytecodeStream *stream,
                   MlirBytecodeBytesRef sectionData[mbci_kNumSections]) {
  uint8_t byte;
  uint64_t length;
  MlirBytecodeStatus ret = mlirBytecodeParseByte(callerState, stream, &byte);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return ret;
  ret = mlirBytecodeParseVarInt(callerState, stream, &length);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return ret;
  uint8_t sectionID = byte & 0x7f;

  mlirBytecodeEmitDebug("parsing %d of %lld", sectionID, length);
  if (MLIRBC_UNLIKELY(sectionID >= mbci_kNumSections))
    return mlirBytecodeEmitError(callerState, "invalid section ID: %d",
                                 sectionID);
  if (MLIRBC_UNLIKELY(sectionData[sectionID].data != NULL)) {
    return mlirBytecodeEmitError(callerState, "duplicate top-level section: %s",
                                 mbci_sectionIDToString(sectionID));
  }

  bool isAligned = byte >> 7;
  if (isAligned) {
    uint64_t alignment;
    ret = mlirBytecodeParseVarInt(callerState, stream, &alignment);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return ret;
    ret = mbci_alignTo(callerState, stream, alignment);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return ret;
  }

  sectionData[sectionID].data = stream->pos;
  sectionData[sectionID].length = length;
  return mbci_skipBytes(callerState, stream, length);
}

MlirBytecodeStatus mlirBytecodeParseNullTerminatedString(
    void *callerState, MlirBytecodeStream *stream, MlirBytecodeBytesRef *str) {
  const uint8_t *startIt = (const uint8_t *)stream->pos;
  const uint8_t *nulIt =
      (const uint8_t *)memchr(startIt, 0, stream->end - stream->pos);
  if (MLIRBC_UNLIKELY(!nulIt)) {
    return mlirBytecodeEmitError(
        callerState,
        "malformed null-terminated string, no null character found");
  }

  str->data = startIt;
  str->length = nulIt - startIt;
  stream->pos += str->length + 1;
  return mlirBytecodeSuccess();
}

static MlirBytecodeStream
mbci_populateStreamForSection(MlirBytecodeBytesRef ref) {
  MlirBytecodeStream stream;
  stream.start = stream.pos = ref.data;
  stream.end = stream.start + ref.length;
  return stream;
}

static MlirBytecodeBytesRef
mbci_getSection(const MlirBytecodeParserState *const file, int index) {
  return file->sectionData[index];
}

MlirBytecodeStatus
mlirBytecodeForEachString(void *callerState,
                          const MlirBytecodeParserState *const parserState) {
  const MlirBytecodeBytesRef stringSection =
      mbci_getSection(parserState, mbci_kString);
  MlirBytecodeStream stream = mbci_populateStreamForSection(stringSection);

  uint64_t numStrings;
  MlirBytecodeStatus ret =
      mlirBytecodeParseVarInt(callerState, &stream, &numStrings);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(callerState,
                                 "failed to parse number of strings");

  // Parse each of the strings. The sizes of the strings are encoded in reverse
  // order, so that's the order we populate the table.
  size_t stringDataEndOffset = stringSection.length;
  for (uint64_t i = numStrings; i > 0; --i) {
    uint64_t stringSize;
    ret = mlirBytecodeParseVarInt(callerState, &stream, &stringSize);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(
          callerState, "failed to parse string size of string %d", i - 1);
    if (MLIRBC_UNLIKELY(stringDataEndOffset < stringSize)) {
      return mlirBytecodeEmitError(
          callerState, "string size exceeds the available data size");
    }

    // Extract the string from the data, dropping the null character.
    size_t stringOffset = stringDataEndOffset - stringSize;
    MlirBytecodeBytesRef str;
    str.data = stringSection.data + stringOffset;
    str.length = stringSize - 1;
    ret = mlirBytecodeStringCallBack(
        callerState, (MlirBytecodeStringHandle){.id = i - 1}, numStrings, str);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(callerState, "string callback failed");
    stringDataEndOffset = stringOffset;
  }

  // Check that the only remaining data was for the strings, i.e. the reader
  // should be at the same offset as the first string.
  if (MLIRBC_UNLIKELY(stream.pos != (stream.start + stringDataEndOffset))) {
    return mlirBytecodeEmitError(
        callerState, "unexpected trailing data between the offsets for strings "
                     "and their data");
  }
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeParseDialectSection(void *callerState,
                                MlirBytecodeParserState *parserState) {
  const MlirBytecodeBytesRef dialectSection =
      mbci_getSection(parserState, mbci_kDialect);
  MlirBytecodeStream stream = mbci_populateStreamForSection(dialectSection);
  mlirBytecodeEmitDebug("parsing dialect section of length %ld",
                        dialectSection.length);

  uint64_t numDialects;
  MlirBytecodeStatus ret =
      mlirBytecodeParseVarInt(callerState, &stream, &numDialects);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(callerState,
                                 "unable to parse number of dialects");
  mlirBytecodeEmitDebug("number of dialects = %d", numDialects);

  uint64_t dialectName;
  for (uint64_t i = 0; i < numDialects; ++i) {
    ret = mlirBytecodeParseVarInt(callerState, &stream, &dialectName);
    if (mlirBytecodeFailed(ret))
      return mlirBytecodeEmitError(callerState, "unable to parse dialect %d",
                                   i);
    mlirBytecodeEmitDebug(
        "dialect[%d] = %s", i,
        mlirBytecodeGetStringSectionValue(
            callerState, (MlirBytecodeDialectHandle){.id = dialectName})
            .data);
    if (mlirBytecodeFailed(mlirBytecodeDialectCallBack(
            callerState, (MlirBytecodeDialectHandle){.id = i}, numDialects,
            (MlirBytecodeStringHandle){.id = dialectName})))
      return mlirBytecodeFailure();
  }

  MlirBytecodeOpHandle op = {.id = 0};
  while (!mbci_streamEmpty(&stream)) {
    uint64_t dialect, numOpNames;
    mlirBytecodeParseVarInt(callerState, &stream, &dialect);
    mlirBytecodeParseVarInt(callerState, &stream, &numOpNames);

    mlirBytecodeEmitDebug("parsing for dialect %d %d ops", dialect, numOpNames);
    for (uint64_t j = 0; j < numOpNames; ++j) {
      uint64_t opName;
      ret = mlirBytecodeParseVarInt(callerState, &stream, &opName);
      if (mlirBytecodeFailed(ret))
        return mlirBytecodeEmitError(callerState, "failed to parse op name");
      mlirBytecodeEmitDebug(
          "\top[%d] = %s (%d) . %s (%d)", (int)op.id,
          mlirBytecodeGetStringSectionValue(
              callerState, (MlirBytecodeDialectHandle){.id = dialect})
              .data,
          dialect,
          mlirBytecodeGetStringSectionValue(
              callerState, (MlirBytecodeStringHandle){.id = opName})
              .data,
          opName);
      // Associate op[index] = {dialect[j], opName}
      ret = mlirBytecodeDialectOpCallBack(
          callerState, op, (MlirBytecodeDialectHandle){.id = dialect},
          (MlirBytecodeStringHandle){.id = opName});
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
      ++op.id;
    }
  }

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeParseAttributeAndTypeSection(void *callerState,
                                         MlirBytecodeParserState *parserState) {
  const MlirBytecodeBytesRef offsetSection =
      mbci_getSection(parserState, mbci_kAttrTypeOffset);
  MlirBytecodeStream offsetStream =
      mbci_populateStreamForSection(offsetSection);
  const MlirBytecodeBytesRef atSection =
      mbci_getSection(parserState, mbci_kAttrType);
  MlirBytecodeStream atPP = mbci_populateStreamForSection(atSection);

  uint64_t numAttrs, numTypes;
  MlirBytecodeStatus ret =
      mlirBytecodeParseVarInt(callerState, &offsetStream, &numAttrs);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(callerState, "invalid number of attributes");
  ret = mlirBytecodeParseVarInt(callerState, &offsetStream, &numTypes);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(callerState,
                                 "invalid number of attributes or types");
  mlirBytecodeEmitDebug("parsing %ld attributes and %ld types", numAttrs,
                        numTypes);

  uint64_t i = 0;
  while (i < numAttrs) {
    uint64_t dialect;
    ret = mlirBytecodeParseVarInt(callerState, &offsetStream, &dialect);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(
          callerState, "invalid dialect handle while parsing attributes");

    uint64_t numElements;
    ret = mlirBytecodeParseVarInt(callerState, &offsetStream, &numElements);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(
          callerState, "invalid number of elements in attr offset group");

    for (uint64_t j = 0; j < numElements; ++j) {
      uint64_t length;
      bool hasCustomEncoding;
      ret = mlirBytecodeParseVarIntWithFlag(callerState, &offsetStream, &length,
                                            &hasCustomEncoding);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(callerState, "invalid attr offset");

      MlirBytecodeBytesRef attr = {.data = atPP.pos, .length = length};

      // Verify that the offset is actually valid.
      if (MLIRBC_UNLIKELY(atPP.pos + length > atPP.end)) {
        return mlirBytecodeEmitError(
            callerState,
            "attribute or Type entry offset points past the end of section");
      }

      // Parse & associate dialect.attr[j] with `attr`
      ret = mlirBytecodeAttrCallBack(
          callerState, (MlirBytecodeAttrHandle){.id = i++}, numAttrs,
          (MlirBytecodeDialectHandle){.id = dialect}, attr, hasCustomEncoding);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret))) {
        mlirBytecodeEmitDebug("attr callback failed");
        return ret;
      }
      if (MLIRBC_UNLIKELY(mlirBytecodeInterrupted(ret)))
        return ret;
      // Unhandled attributes are not considered error.

      if (mlirBytecodeFailed(mbci_skipBytes(callerState, &atPP, length)))
        return mlirBytecodeEmitError(callerState, "invalid attr offset");
    }
  }

  i = 0;
  while (i < numTypes) {
    uint64_t dialect;
    ret = mlirBytecodeParseVarInt(callerState, &offsetStream, &dialect);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(
          callerState, "invalid dialect handle while parsing types");

    uint64_t numElements;
    ret = mlirBytecodeParseVarInt(callerState, &offsetStream, &numElements);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(
          callerState, "invalid number of elements in type offset group");

    for (uint64_t j = 0; j < numElements; ++j) {
      uint64_t offset;
      bool hasCustomEncoding;
      ret = mlirBytecodeParseVarIntWithFlag(callerState, &offsetStream, &offset,
                                            &hasCustomEncoding);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(callerState, "invalid type offset");

      MlirBytecodeBytesRef type = {.data = atPP.pos, .length = offset};
      // Verify that the offset is actually valid.
      if (MLIRBC_UNLIKELY(atPP.pos + offset > atPP.end)) {
        return mlirBytecodeEmitError(
            callerState,
            "Attribute or Type entry offset points past the end of section");
      }

      ret = mlirBytecodeTypeCallBack(
          callerState, (MlirBytecodeTypeHandle){.id = i++}, numTypes,
          (MlirBytecodeDialectHandle){.id = dialect}, type, hasCustomEncoding);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret))) {
        mlirBytecodeEmitDebug("attr callback failed");
        return ret;
      }
      if (MLIRBC_UNLIKELY(mlirBytecodeInterrupted(ret)))
        return ret;
      // Unhandled attributes are not considered error.

      ret = mbci_skipBytes(callerState, &atPP, offset);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(callerState, "invalid type offset");
    }
  }

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeParseResourceSection(void *callerState,
                                 MlirBytecodeParserState *const parserState) {

  enum MlirBytecodeAsmResourceEntryKind {
    /// A blob of data with an accompanying alignment.
    kMlirBytecodeResourceEntryBlob,
    /// A boolean value.
    kMlirBytecodeResourceEntryBool,
    /// A string value.
    kMlirBytecodeResourceEntryString,
  };
  typedef enum MlirBytecodeAsmResourceEntryKind
      MlirBytecodeAsmResourceEntryKind;

  const MlirBytecodeBytesRef offsetSection =
      mbci_getSection(parserState, mbci_kResourceOffset);
  MlirBytecodeStream offsetStream =
      mbci_populateStreamForSection(offsetSection);
  const MlirBytecodeBytesRef resSection =
      mbci_getSection(parserState, mbci_kResourceOffset);
  MlirBytecodeStream resPP = mbci_populateStreamForSection(resSection);

  uint64_t numExternalResourceGroups;
  MlirBytecodeStatus ret = mlirBytecodeParseVarInt(callerState, &offsetStream,
                                                   &numExternalResourceGroups);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(
        callerState, "invalid/missing number of external resource groups");
  if (!numExternalResourceGroups)
    return mlirBytecodeSuccess();

  ret =
      mlirBytecodeResourceSectionEnter(callerState, numExternalResourceGroups);
  if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
    return ret;

  for (uint64_t i = 0; i < numExternalResourceGroups; ++i) {
    MlirBytecodeStringHandle groupKey;
    ret = mlirBytecodeParseHandle(callerState, &offsetStream, &groupKey);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(callerState, "invalid/missing "
                                                "resource group key");
    mlirBytecodeEmitDebug(
        "Group for %s / %d",
        mlirBytecodeGetStringSectionValue(callerState, groupKey).data,
        numExternalResourceGroups);

    uint64_t numResources;
    ret = mlirBytecodeParseVarInt(callerState, &offsetStream, &numResources);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(callerState,
                                   "invalid/missing number of resources");

    ret = mlirBytecodeResourceGroupEnter(callerState, groupKey, numResources);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;

    for (uint64_t j = 0; j < numResources; ++j) {
      uint64_t resourceKey;
      ret = mlirBytecodeParseVarInt(callerState, &offsetStream, &resourceKey);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(callerState,
                                     "invalid/missing resource key");
      uint64_t size;
      ret = mlirBytecodeParseVarInt(callerState, &offsetStream, &size);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(callerState,
                                     "invalid/missing resource size");
      uint8_t kind;
      ret = mlirBytecodeParseByte(callerState, &offsetStream, &kind);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(callerState,
                                     "invalid/missing resource kind");

      MlirBytecodeAsmResourceEntryKind resKind =
          (MlirBytecodeAsmResourceEntryKind)kind;
      switch (resKind) {
      case kMlirBytecodeResourceEntryBlob: {
        uint64_t alignment;
        ret = mlirBytecodeParseVarInt(callerState, &resPP, &alignment);
        if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
          return mlirBytecodeEmitError(callerState,
                                       "invalid/missing resource alignment");
        uint64_t size;
        ret = mlirBytecodeParseVarInt(callerState, &resPP, &size);
        if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
          return mlirBytecodeEmitError(callerState,
                                       "invalid/missing resource size");
        ret = mbci_alignTo(callerState, &resPP, alignment);
        if (mlirBytecodeFailed(ret))
          return mlirBytecodeFailure();
        MlirBytecodeBytesRef blob = {.data = resPP.pos, .length = size};
        ret = mlirBytecodeResourceBlobCallBack(
            callerState, groupKey,
            (MlirBytecodeStringHandle){.id = resourceKey}, blob);
        if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
          return ret;
        break;
      }
      case kMlirBytecodeResourceEntryBool: {
        uint8_t value;
        ret = mlirBytecodeParseByte(callerState, &resPP, &value);
        if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
          return mlirBytecodeEmitError(callerState,
                                       "invalid/missing resource entry");
        ret = mlirBytecodeResourceBoolCallBack(
            callerState, groupKey,
            (MlirBytecodeStringHandle){.id = resourceKey}, value);
        if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
          return ret;
        break;
      }
      case kMlirBytecodeResourceEntryString: {
        MlirBytecodeStringHandle value;
        ret = mlirBytecodeParseHandle(callerState, &resPP, &value);
        if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
          return mlirBytecodeEmitError(callerState,
                                       "invalid/missing resource value");
        ret = mlirBytecodeResourceStringCallBack(
            callerState, groupKey,
            (MlirBytecodeStringHandle){.id = resourceKey}, value);
        if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
          return ret;
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

  mbci_MlirIRSectionStackEntry data[MLIRBC_IR_STACK_MAX_DEPTH + 1];
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
mbci_mlirIRSectionStackPush(void *callerState, mbci_MlirIRSectionStack *stack) {
  if (MLIRBC_UNLIKELY(stack->top == MLIRBC_IR_STACK_MAX_DEPTH)) {
    mlirBytecodeEmitError(callerState, "IR stack max depth exceeded");
    return mlirBytecodeIterationInterrupt();
  }
  ++stack->top;
  mbci_MlirIRSectionStackEntry *it = mbci_getStackTop(stack);
  it->numBlocksRemaining = mbci_MlirIRSectionStackUnset;
  it->numOpsRemaining = mbci_MlirIRSectionStackUnset;

  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus
mbci_mlirIRSectionStackPop(void *callerState, mbci_MlirIRSectionStack *stack) {
  --stack->top;
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus mbci_parseBlock(void *callerState,
                                          MlirBytecodeStream *stream,
                                          mbci_MlirIRSectionStack *stack) {
  mbci_MlirIRSectionStackEntry *top = mbci_getStackTop(stack);
  bool hasArgs;
  uint64_t numOpsRemaining;
  MlirBytecodeStatus ret = mlirBytecodeParseVarIntWithFlag(
      callerState, stream, &numOpsRemaining, &hasArgs);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(callerState,
                                 "unable to parse block's number of args");
  top->numOpsRemaining = numOpsRemaining;
  mlirBytecodeEmitDebug("num ops = %d and block %s", numOpsRemaining,
                        hasArgs ? "has args" : "has no args");

  uint64_t numArgs = 0;
  if (hasArgs) {
    ret = mlirBytecodeParseVarInt(callerState, stream, &numArgs);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(callerState,
                                   "invalid/missing number of block args");
  }

  ret = mlirBytecodeOperationStateBlockPush(callerState, top->op, stream,
                                            numArgs);
  return ret;
}

static MlirBytecodeStatus mbci_parseRegion(void *callerState,
                                           MlirBytecodeStream *stream,
                                           mbci_MlirIRSectionStack *stack,
                                           void *state) {
  mbci_MlirIRSectionStackEntry *top = mbci_getStackTop(stack);
  uint64_t numBlocksRemaining;
  MlirBytecodeStatus ret =
      mlirBytecodeParseVarInt(callerState, stream, &numBlocksRemaining);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(callerState, "invalid number of blocks");
  top->numBlocksRemaining = numBlocksRemaining;
  top->numOpsRemaining = mbci_MlirIRSectionStackUnset;
  if (top->numBlocksRemaining == 0)
    return mlirBytecodeSuccess();

  uint64_t numValues;
  ret = mlirBytecodeParseVarInt(callerState, stream, &numValues);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(callerState,
                                 "invalid/missing number of values in region");
  mlirBytecodeEmitDebug("numBlocks = %d numValues = %d", numBlocksRemaining,
                        numValues);

  return mlirBytecodeRegionPush(state, top->op, top->numBlocksRemaining,
                                numValues);
}

static MlirBytecodeStatus
mbci_parseOperation(void *callerState, MlirBytecodeParserState *parserState,
                    MlirBytecodeStream *stream,
                    mbci_MlirIRSectionStack *stack) {
  enum OpEncodingMask {
    kHasAttrs = 0x01,
    kHasResults = 0x02,
    kHasOperands = 0x04,
    kHasSuccessors = 0x08,
    kHasInlineRegions = 0x10,
  };

  MlirBytecodeOpHandle name;
  MlirBytecodeStatus ret =
      mlirBytecodeParseVarInt(callerState, stream, &name.id);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(callerState, "invalid operation");

  uint8_t encodingMask;
  ret = mlirBytecodeParseByte(callerState, stream, &encodingMask);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(callerState, "invalid encoding mask");

  MlirBytecodeLocHandle loc;
  ret = mlirBytecodeParseVarInt(callerState, stream, &loc.id);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(callerState, "invalid operation location");

  mbci_MlirIRSectionStackEntry *cur = mbci_getStackWIP(stack);
  ret = mlirBytecodeOperationStatePush(callerState, name, loc, &cur->op);
  if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
    return ret;

  if (encodingMask & kHasAttrs) {
    MlirBytecodeAttrHandle attrDict;
    ret = mlirBytecodeParseHandle(callerState, stream, &attrDict);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret

                                           ))) {
      return mlirBytecodeEmitError(callerState, "invalid op attribute handle");
    }
    ret = mlirBytecodeOperationStateAddAttributeDictionary(callerState, cur->op,
                                                           attrDict);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return mlirBytecodeEmitDebug("unable to add attributes"), ret;
  }

  // Parsing all the variadic sizes.
  if (encodingMask & kHasResults) {
    uint64_t numResults = 0;
    ret = mlirBytecodeParseVarInt(callerState, stream, &numResults);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
    MlirBytecodeHandle* range;
    mbci_pushOnScratch(parserState, numResults, &range);
    for (MlirBytecodeSize i = 0; i < numResults; ++i) {
      ret = mlirBytecodeParseHandle(callerState, stream, &range[i]);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
    }
    ret = mlirBytecodeOperationStateAddResultTypes(
        callerState, cur->op,
        (MlirBytecodeHandlesRef){.handles = range,
                                 .length = numResults});
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
  }

  if (encodingMask & kHasOperands) {
    uint64_t numOperands = 0;
    ret = mlirBytecodeParseVarInt(callerState, stream, &numOperands);
    if (MLIRBC_UNLIKELY(mlirBytecodeSucceeded(ret))) {
      ret = mlirBytecodeOperationStateAddOperands(callerState, cur->op, stream,
                                                  numOperands);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return mlirBytecodeEmitError(callerState, "invalid operands"), ret;
    } else {
      return mlirBytecodeEmitError(callerState, "invalid number of operands");
    }
  }

  if (encodingMask & kHasSuccessors) {
    uint64_t numSuccessors = 0;
    ret = mlirBytecodeParseVarInt(callerState, stream, &numSuccessors);
    if (MLIRBC_UNLIKELY(mlirBytecodeSucceeded(ret))) {
      ret = mlirBytecodeOperationStateAddSuccessors(callerState, cur->op,
                                                    stream, numSuccessors);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return mlirBytecodeEmitError(callerState, "invalid successors"), ret;
    } else {
      return mlirBytecodeEmitError(callerState, "invalid number of successors");
    }
  }

  if (encodingMask & kHasInlineRegions) {
    uint64_t numRegions = 0;
    bool isIsolatedFromAbove = false;
    ret = mlirBytecodeParseVarIntWithFlag(callerState, stream, &numRegions,
                                          &isIsolatedFromAbove);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret))) {
      return mlirBytecodeEmitError(callerState, "invalid number of regions");
    }
    cur->numRegionsRemaining = numRegions;
    cur->isIsolatedFromAbove = isIsolatedFromAbove;
    ret =
        mlirBytecodeOperationStateAddRegions(callerState, cur->op, numRegions);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
    return mbci_mlirIRSectionStackPush(callerState, stack);
  }

  // Completed parsing of op.
  ret = mlirBytecodeOperationStatePop(callerState, cur->op);
  if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
    return ret;
  // `cur` is the scratchpad for parsing op without region while `top` is where
  // it is being parsed.
  --mbci_getStackTop(stack)->numOpsRemaining;
  return ret;
}

MlirBytecodeStatus
mlirBytecodeParseIRSection(void *callerState,
                           MlirBytecodeParserState *parserState) {
  const MlirBytecodeBytesRef irSection = mbci_getSection(parserState, mbci_kIR);
  MlirBytecodeStream stream = mbci_populateStreamForSection(irSection);

  bool hasArgs;
  uint64_t numOps;
  MlirBytecodeStatus ret =
      mlirBytecodeParseVarIntWithFlag(callerState, &stream, &numOps, &hasArgs);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeFailure();
  if (MLIRBC_UNLIKELY(hasArgs))
    return mlirBytecodeEmitError(callerState,
                                 "IR section may not have block args");
  if (MLIRBC_UNLIKELY(numOps != 1))
    return mlirBytecodeEmitError(callerState,
                                 "only one top-level op supported");

  // Initialize with parsing of top-level.
  mbci_MlirIRSectionStack stack = {.top = -1};
  mbci_parseOperation(callerState, parserState, &stream, &stack);

  while (stack.top != -1) {
    mbci_MlirIRSectionStackEntry *top = mbci_getStackTop(&stack);

    MlirBytecodeStatus ret;
    bool next = false;
    if (top->numOpsRemaining == 0) {
      mlirBytecodeEmitDebug("no ops remaining in block");
      ret = mlirBytecodeOperationStateBlockPop(callerState, top->op);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;

      --top->numBlocksRemaining;
      top->numOpsRemaining = mbci_MlirIRSectionStackUnset;
      next = true;
    }
    if (top->numBlocksRemaining == 0) {
      mlirBytecodeEmitDebug("no blocks remaining in region");
      ret = mlirBytecodeOperationStateRegionPop(callerState, top->op);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;

      --top->numRegionsRemaining;
      top->numBlocksRemaining = mbci_MlirIRSectionStackUnset;
      next = true;
    }
    if (top->numRegionsRemaining == 0) {
      mlirBytecodeEmitDebug("no regions remaining in op");
      ret = mlirBytecodeOperationStatePop(callerState, top->op);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;

      mbci_mlirIRSectionStackPop(callerState, &stack);
      --mbci_getStackTop(&stack)->numOpsRemaining;
      next = true;
    }
    if (next)
      continue;

    if (top->numBlocksRemaining == (uint16_t)mbci_MlirIRSectionStackUnset) {
      mlirBytecodeEmitDebug("starting region parsing");
      ret = mbci_parseRegion(callerState, &stream, &stack, callerState);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
      continue;
    }
    if (top->numOpsRemaining == (uint32_t)mbci_MlirIRSectionStackUnset) {
      mlirBytecodeEmitDebug("starting block parsing");
      ret = mbci_parseBlock(callerState, &stream, &stack);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
      continue;
    }
    mlirBytecodeEmitDebug(
        "parsing operation (remaining regions: %d blocks: %d ops: %d)",
        (int)mbci_getStackTop(&stack)->numRegionsRemaining,
        (int)mbci_getStackTop(&stack)->numBlocksRemaining,
        (int)mbci_getStackTop(&stack)->numOpsRemaining);
    ret = mbci_parseOperation(callerState, parserState, &stream, &stack);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
  }
  return mlirBytecodeSuccess();
}

bool mlirBytecodeParserStateEmpty(MlirBytecodeParserState *file) {
  return file->producer.length == 0;
}

MlirBytecodeParserState
mlirBytecodePopulateParserState(void *callerState,
                                const MlirBytecodeBytesRef bytes) {
  MlirBytecodeParserState ret;
  memset(&ret, 0, sizeof(MlirBytecodeParserState));
  MlirBytecodeStream stream = mbci_populateStreamForSection(bytes);

  const uint8_t *magic;
  if (mlirBytecodeFailed(
          mlirBytecodeParseBytes(callerState, &stream, 4, &magic)))
    return mlirBytecodeEmitError(callerState,
                                 "unable to read 4 byte magic code"),
           ret;
  if (magic[0] != 'M' || magic[1] != 'L' || magic[2] != 0xef || magic[3] != 'R')
    return mlirBytecodeEmitError(callerState, "invalid file magic code"), ret;

  // Parse the bytecode version and producer.
  MlirBytecodeBytesRef producer;
  if (mlirBytecodeFailed(
          mlirBytecodeParseVarInt(callerState, &stream, &ret.version)) ||
      mlirBytecodeFailed(mlirBytecodeParseNullTerminatedString(
          callerState, &stream, &producer)))
    return mlirBytecodeEmitError(callerState, "invalid version or producer"),
           ret;
  mlirBytecodeEmitDebug("bytecode producer: %s\n", producer.data);

  while (!mbci_streamEmpty(&stream)) {
    // Read the next section from the bytecode.
    if (mlirBytecodeFailed(
            mbci_parseSections(callerState, &stream, ret.sectionData)))
      return mlirBytecodeEmitError(callerState, "invalid section"), ret;
  }

  // Check that all of the required sections were found
  for (int i = 0; i < mbci_kNumSections; ++i) {
    if (mbci_getSection(&ret, i).data == NULL && !mbci_isSectionOptional(i)) {
      return mlirBytecodeEmitError(callerState,
                                   "missing data for top-level section: ",
                                   mbci_sectionIDToString(i)),
             ret;
    }
  }

  // TODO: Allow passing in.
  ret.scratchBase = 0;
  ret.capacity = ret.size = 0;

  // Mark as valid.
  ret.producer = producer;
  return ret;
}

MlirBytecodeStatus mlirBytecodeParse(void *callerState,
                                     MlirBytecodeBytesRef bytes) {
  MlirBytecodeParserState parserState =
      mlirBytecodePopulateParserState(callerState, bytes);
  if (mlirBytecodeParserStateEmpty(&parserState))
    return mlirBytecodeFailure();

  // Process the string section first.
  if (mlirBytecodeFailed(
          mlirBytecodeForEachString(callerState, &parserState)) ||
      // Process the dialect section.
      mlirBytecodeFailed(
          mlirBytecodeParseDialectSection(callerState, &parserState)) ||
      // Process the resource section if present.
      mlirBytecodeFailed(
          mlirBytecodeParseResourceSection(callerState, &parserState)) ||
      // Process the attribute and type section.
      mlirBytecodeFailed(mlirBytecodeParseAttributeAndTypeSection(
          callerState, &parserState)) ||
      // Finally, process the IR section.
      mlirBytecodeFailed(mlirBytecodeParseIRSection(callerState, &parserState)))
    return mlirBytecodeFailure();

  return mlirBytecodeSuccess();
}
