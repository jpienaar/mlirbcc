//===-- Parse.h - MLIR bytecode C parser --------------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// C API for MLIR bytecode event-driven parser.
//===----------------------------------------------------------------------===//

// This file contains the full implementation for an event-based MLIR bytecode
// parser. It is defined as a pure header-based implementation. To use:
//
//   // Define MlirBytecodeOperationState that is kept on stack during parsing
//   // operation with regions.
//   #define MLIRBC_PARSE_IMPLEMENTATION
//   #include "mlirbcc/Parse.h"
//   // Include dialect specific parsing extensions with required types.
//   #include "mlirbcc/BuiltinParse.h"
//
// Define types and functions required for parsing (see below).
//
// Callbacks/functions to be implemented in instantiation accept an opaque
// pointer as first argument that gets directly propagated during parsing and
// can be used by instantiation for additional/parse local state capture.
//
// Note: only set implementation define before you include this file in one C or
// C++ file.

// Convention followed in this file is to prefix implementation details with
// mbci_.

#ifndef MLIRBC_PARSE_H
#define MLIRBC_PARSE_H

#include "mlirbcc/BytecodeTypes.h"
#include "mlirbcc/Log.h"
#include "mlirbcc/Status.h"

#include <stdalign.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Indicates functions users
#define MLIRBC_DEF extern

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Functions and types required:
// - Types
//      MlirBytecodeOperationState;

// - Functions

/// Called when creating variable to populate operation state in.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeOperationStatePush(void *callerState, MlirBytecodeOperationState *);
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeOperationStateAddAttributes(void* callerState, MlirBytecodeOperationState* cur, MlirBytecodeAttrHandle attrDict);
/// Called when finalizing population of operation state.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeOperationStatePop(void *callerState, MlirBytecodeOperationState *);

/// Called when entering a region with numBlocks blocks and numValues Values
/// (including values due to block args).
typedef MlirBytecodeStatus (*MlirBytecodeRegionEnter)(void *, bool isIsolated,
                                                      size_t numBlocks,
                                                      size_t numValues);

/// Called when entering block in region. The blockArgs consists of pair
/// type and location and numOps is the number of ops in the block.
typedef MlirBytecodeStatus (*MlirBytecodeBlockEnter)(
    void *, MlirBlockArgHandleIterator *blockArgs, size_t numOps);

/// Called per operation in block.
typedef MlirBytecodeStatus (*MlirBytecodeOperation)(
    void *, MlirBytecodeOpHandle name, MlirBytecodeAttrHandle attrDict,
    MlirBytecodeStream *resultTypes, MlirBytecodeStream *operands,
    MlirBytecodeStream *successors, bool isIsolatedFromAbove,
    size_t numRegions);

/// Called when exiting the block.
typedef MlirBytecodeStatus (*MlirBytecodeBlockExit)(void *);

/// Called when entering a region.
typedef MlirBytecodeStatus (*MlirBytecodeRegionExit)(void *, bool isIsolated);

/// Called post completed parsing of an isolated from above operation.
typedef MlirBytecodeStatus (*MlirBytecodeIsolatedOperationExit)(
    void *, bool isIsolated);

//===----------------------------------------------------------------------===//
// Section parsing entry points.

/// Associate dialect and opname with string handle.
typedef MlirBytecodeStatus (*MlirBytecodeDialectOpCallBack)(
    void *, MlirBytecodeDialectHandle, MlirBytecodeOpHandle,
    MlirBytecodeStringHandle);

/// Associate dialect attribute with range in memory.
typedef MlirBytecodeStatus (*MlirBytecodeAttrCallBack)(
    void *, MlirBytecodeDialectHandle, MlirBytecodeAttrHandle, size_t /*total*/,
    bool, MlirBytecodeBytesRef);

/// Associate dialect type with range in memory.
typedef MlirBytecodeStatus (*MlirBytecodeTypeCallBack)(
    void *, MlirBytecodeDialectHandle, MlirBytecodeTypeHandle, size_t /*total*/,
    bool, MlirBytecodeBytesRef);

/// Associate groupKey[resourceKey] with either MlirBytecodeStatus, string or
/// blob.
typedef MlirBytecodeStatus (*MlirBytecodeResourceCallBack)(
    void *, MlirBytecodeStringHandle groupKey, MlirBytecodeSize totalGroups,
    MlirBytecodeStringHandle resourceKey, MlirBytecodeAsmResourceEntryKind,
    const MlirBytecodeBytesRef *blob, const uint8_t *MlirBytecodeStatusResource,
    const MlirBytecodeStringHandle *);

/// String callback which consists of string handle, total number of strings in
/// string section and bytes corresponding to the string.
typedef MlirBytecodeStatus (*MlirBytecodeStringCallBack)(
    void *, MlirBytecodeStringHandle, size_t /*total*/, MlirBytecodeBytesRef);

/// Iterators over attributes and types, calling MlirBytecodeAttrCallBack and
/// MlirBytecodeTypeCallBack upon encountering Attribute or Type respectively.
/// Returns whether failed.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeForEachAttributeAndType(
    void *callerState, MlirBytecodeFile *mlirFile,
    MlirBytecodeAttrCallBack attrFn, MlirBytecodeTypeCallBack typeFn);

/// Associate dialect with string handle.
typedef MlirBytecodeStatus (*MlirBytecodeDialectCallBack)(
    void *, MlirBytecodeDialectHandle, size_t /*total*/,
    MlirBytecodeStringHandle);

/// Parses the dialect section, invoking MlirBytecodeDialectCallBack upon
/// dialect encountered and MlirBytecodeDialectOpCallBack per operation type in
/// dialect. Returns whether failed.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeParseDialectSection(
    void *callerState, MlirBytecodeFile *mlirFile,
    MlirBytecodeDialectCallBack dialectFn, MlirBytecodeDialectOpCallBack opFn);

/// Parse IR section in mlirFile. The block args, operation and region callback
/// are invoked during bytecode in-order walk. Additionally allows for passing
/// in an opaque state.
///
/// The IR section parsing follows the nesting order:
///   op ->* regions ->* blocks
/// The caller is required to keep track of when all operations/blocks in
/// block/region have been processed and so parsing resumes at parent level.
/// Returns whether failed.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeParseIRSection(
    void *callerState, MlirBytecodeFile *mlirFile, MlirBytecodeOperation opFn,
    MlirBytecodeRegionEnter regionEnterFn, MlirBytecodeBlockEnter blockEnterFn,
    MlirBytecodeBlockExit blockExitFn, MlirBytecodeRegionExit regionExitFn,
    MlirBytecodeIsolatedOperationExit opExitFn);

/// Parse the resource section, calling MlirBytecodeResourceCallBack upon
/// resources encountered. Returns whether failed.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeForEachResource(void *callerState, MlirBytecodeFile *mlirFile,
                            MlirBytecodeResourceCallBack fn);

/// Invoke the callback per string in string section.
/// Returns whether failed.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeForEachString(
    void *callerState, const MlirBytecodeFile *const mlirFile,
    MlirBytecodeStringCallBack fn);

//===----------------------------------------------------------------------===//
// MLIR file parsing methods.

/// Populates the MlirBytecodeFile contents for given in memory bytes.
/// Returns an empty file if population failed.
MLIRBC_DEF MlirBytecodeFile
mlirBytecodePopulateFile(MlirBytecodeBytesRef bytes);

/// Parses the given MLIR file represented in memory `bytes`, calls the
/// appropriate callbacks during parsing. This combines the parsing methods
/// below.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeParseFile(
    void *callerState, MlirBytecodeBytesRef bytes,
    MlirBytecodeAttrCallBack attrFn, MlirBytecodeTypeCallBack typeFn,
    MlirBytecodeDialectCallBack dialectFn,
    MlirBytecodeDialectOpCallBack dialectOpFn,
    MlirBytecodeResourceCallBack resourceFn,
    MlirBytecodeStringCallBack stringFn,
    // IR section parsing callbacks.
    MlirBytecodeOperation opFn, MlirBytecodeRegionEnter regionEnterFn,
    MlirBytecodeBlockEnter blockEnterFn, MlirBytecodeBlockExit blockExitFn,
    MlirBytecodeRegionExit regionExitFn,
    MlirBytecodeIsolatedOperationExit opExitFn);

/// Returns whether the given MlirBytecodeFile structure is empty.
MLIRBC_DEF bool mlirBytecodeFileEmpty(MlirBytecodeFile *file);

/// Returns whether the given attribute is the sentinel value.
MLIRBC_DEF bool mlirBytecodeIsSentinel(MlirBytecodeAttrHandle attr);

//===----------------------------------------------------------------------===//
// Dialect parsing utility methods.
// Note: These should probably go into their own header.

/// Creates a bytecode stream from section of bytes.
MLIRBC_DEF MlirBytecodeStream
mlirBytecodeStreamCreate(MlirBytecodeBytesRef ref);

/// Reset the stream to its head.
MLIRBC_DEF void mlirBytecodeStreamReset(MlirBytecodeStream *iterator);

/// Decode the next handle on the stream and increment stream.
MlirBytecodeStatus mlirBytecodeReadAttrHandle(MlirBytecodeStream *iterator,
                                             MlirBytecodeAttrHandle *result);

/// Decode uint64 on the stream and increment stream.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeReadVarInt(MlirBytecodeStream *iterator, uint64_t *result);

/// Decode int64 on the stream and increment stream.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeReadSignedVarInt(MlirBytecodeStream *iterator, int64_t *result);

/// Parse a variable length encoded integer whose low bit is used to encode a
/// flag, i.e: `(integerValue << 1) | (flag ? 1 : 0)`.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeParseVarIntWithFlag(
    MlirBytecodeStream *pp, uint64_t *result, bool *flag);

/// Parse a null-terminated string.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeParseNullTerminatedString(
    MlirBytecodeStream *pp, MlirBytecodeBytesRef *str);

/// Populate the next block arg type & location and increment the iterator.
/// Returns whether there was an element.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeGetNextBlockArgHandles(
    MlirBlockArgHandleIterator *iterator, MlirBytecodeTypeHandle *,
    MlirBytecodeAttrHandle *);

//===----------------------------------------------------------------------===//
// Lazy parsing methods
// Note: these methods don't cache any state by default. They can be overridden
// by instantiations.

// Returns the requested string from the string section.
MLIRBC_DEF MlirBytecodeBytesRef mlirBytecodeGetStringSectionValue(
    void *callerState, const MlirBytecodeFile *mlirFile,
    MlirBytecodeStringHandle idx);

// Returns the requested dialect & operation for given index.
// Note: this doesn't cache any state.
MLIRBC_DEF MlirBytecodeOpRef
mlirBytecodeGetOpName(void *callerState, const MlirBytecodeFile *mlirFile,
                      MlirBytecodeOpHandle hdl);

#ifdef __cplusplus
}
#endif
#endif // MLIRBC_PARSE_H

#ifdef MLIRBC_PARSE_IMPLEMENTATION
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

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

struct MlirFileInternal {
  uint64_t version;
  MlirBytecodeBytesRef producer;

  MlirBytecodeBytesRef sectionData[mbci_kNumSections];
};
typedef struct MlirFileInternal MlirFileInternal;

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
__attribute__((weak)) MlirBytecodeStatus
mlirBytecodeEmitErrorImpl(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");

  return mlirBytecodeFailure();
}
#else
__attribute__((weak)) MlirBytecodeStatus
mlirBytecodeEmitErrorImpl(const char *fmt, ...) {
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
__attribute__((weak)) void mlirBytecodeEmitDebugImpl(const char *file, int line,
                                                     const char *fmt, ...) {
  (void)fmt;
}
#endif

//-----
// Base parsing primitives.
//=====

// Represents simple parser state.
static bool mbci_streamEmpty(MlirBytecodeStream *pp) {
  return pp->pos >= pp->end;
}

static MlirBytecodeStatus mbci_parseByte(MlirBytecodeStream *pp, uint8_t *val) {
  if (mbci_streamEmpty(pp))
    return mlirBytecodeFailure();
  *val = *pp->pos++;
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus mbci_parseBytes(MlirBytecodeStream *pp, uint8_t *val,
                                          size_t n) {
  for (uint64_t i = 0; i < n; ++i)
    if (mlirBytecodeFailed(mbci_parseByte(pp, val++)))
      return mlirBytecodeFailure();
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus mbci_skipBytes(MlirBytecodeStream *pp, size_t n) {
  pp->pos += n;
  if (pp->pos > pp->end)
    return mlirBytecodeFailure();
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus mbci_parseVarInt(MlirBytecodeStream *pp,
                                           uint64_t *result) {
  // Compute the number of bytes needed to encode the value. Each byte can hold
  // up to 7-bits of data. We only check up to the number of bits we can encode
  // in the first byte (8).
  uint8_t head;
  if (mlirBytecodeFailed(mbci_parseByte(pp, &head)))
    return mlirBytecodeFailure();
  *result = head;
  if ((*result & 1)) {
    *result >>= 1;
    return mlirBytecodeSuccess();
  }

  if (*result == 0) {
    uint8_t *ptr = (uint8_t *)(result);
    if (mlirBytecodeFailed(mbci_parseBytes(pp, ptr, sizeof(*result))))
      return mlirBytecodeFailure();
    return mlirBytecodeSuccess();
  }

  // Parse in the remaining bytes of the value.
  uint32_t numBytes = __builtin_ctz(*result);
  uint8_t *ptr = (uint8_t *)(result) + 1;
  if (mlirBytecodeFailed(mbci_parseBytes(pp, ptr, numBytes)))
    return mlirBytecodeFailure();

  // Shift out the low-order bits that were used to mark how the value was
  // encoded.
  *result >>= (numBytes + 1);
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus mbci_skipVarInts(MlirBytecodeStream *pp, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    uint8_t head;
    if (mlirBytecodeFailed(mbci_parseByte(pp, &head)))
      return mlirBytecodeFailure();
    int numBytes = (head == 0) ? 8 : __builtin_ctz(head);
    if (mlirBytecodeFailed(mbci_skipBytes(pp, numBytes)))
      return mlirBytecodeFailure();
  }
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeReadAttrHandle(MlirBytecodeStream *iterator,
                                             MlirBytecodeAttrHandle *result) {
  return mlirBytecodeReadVarInt(iterator, &result->id);
}

void mlirBytecodeStreamReset(MlirBytecodeStream *iterator) {
  iterator->pos = iterator->start;
}

MlirBytecodeStatus mlirBytecodeReadVarInt(MlirBytecodeStream *iterator,
                                          uint64_t *result) {
  uint64_t ret;
  if (mlirBytecodeFailed(mbci_parseVarInt(iterator, &ret)))
    return mlirBytecodeFailure();
  *result = ret;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeGetNextBlockArgHandles(MlirBlockArgHandleIterator *iterator,
                                   MlirBytecodeTypeHandle *type,
                                   MlirBytecodeAttrHandle *loc) {
  if (mbci_streamEmpty((MlirBytecodeStream *)iterator))
    return mlirBytecodeFailure();
  if (mlirBytecodeFailed(
          mbci_parseVarInt((MlirBytecodeStream *)iterator, (uint64_t *)type)) ||
      mlirBytecodeFailed(
          mbci_parseVarInt((MlirBytecodeStream *)iterator, (uint64_t *)loc)))
    return mlirBytecodeFailure();
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeParseVarIntWithFlag(MlirBytecodeStream *pp,
                                                   uint64_t *result,
                                                   bool *flag) {
  if (mlirBytecodeFailed(mbci_parseVarInt(pp, result)))
    return mlirBytecodeFailure();
  *flag = *result & 1;
  *result >>= 1;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeReadSignedVarInt(MlirBytecodeStream *pp,
                                                int64_t *result) {
  if (mlirBytecodeFailed(mbci_parseVarInt(pp, (uint64_t *)result)))
    return mlirBytecodeFailure();
  *result = (*result >> 1) ^ (~(*result & 1) + 1);
  return mlirBytecodeSuccess();
}

// Align the current reader position to the specified alignment.
static MlirBytecodeStatus mbci_alignTo(MlirBytecodeStream *pp,
                                       uint32_t alignment) {
  bool isPowerOf2 = (alignment != 0) && ((alignment & (alignment - 1)) == 0);
  if (!isPowerOf2)
    return mlirBytecodeEmitError("expected alignment to be a power-of-two");

  // An arbitrary value used to fill alignment padding.
  const uint8_t kAlignmentByte = 0xCB;

  // Shift the reader position to the next alignment boundary.
  while ((uintptr_t)pp->pos & ((uintptr_t)alignment - 1)) {
    uint8_t padding;
    if (mlirBytecodeFailed(mbci_parseByte(pp, &padding)))
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
mbci_parseSections(MlirBytecodeStream *pp,
                   MlirBytecodeBytesRef sectionData[mbci_kNumSections]) {
  uint8_t byte;
  uint64_t length;
  if (mlirBytecodeFailed(mbci_parseByte(pp, &byte)) ||
      mlirBytecodeFailed(mbci_parseVarInt(pp, &length)))
    return mlirBytecodeFailure();
  uint8_t sectionID = byte & 0x7f;

  mlirBytecodeEmitDebug("Parsing %d of %lld", sectionID, length);
  if (sectionID >= mbci_kNumSections)
    return mlirBytecodeEmitError("invalid section ID: %d", sectionID);
  if (sectionData[sectionID].data != NULL) {
    return mlirBytecodeEmitError("duplicate top-level section: %s",
                                 mbci_sectionIDToString(sectionID));
  }

  bool isAligned = byte >> 7;
  if (isAligned) {
    uint64_t alignment;
    if (mlirBytecodeFailed(mbci_parseVarInt(pp, &alignment)) ||
        mlirBytecodeFailed(mbci_alignTo(pp, alignment)))
      return mlirBytecodeFailure();
  }

  sectionData[sectionID].data = pp->pos;
  sectionData[sectionID].length = length;
  return mbci_skipBytes(pp, length);
}

MlirBytecodeStatus
mlirBytecodeParseNullTerminatedString(MlirBytecodeStream *pp,
                                      MlirBytecodeBytesRef *str) {
  const uint8_t *startIt = (const uint8_t *)pp->pos;
  const uint8_t *nulIt = (const uint8_t *)memchr(startIt, 0, pp->end - pp->pos);
  if (!nulIt) {
    return mlirBytecodeEmitError(
        "malformed null-terminated string, no null character found");
  }

  str->data = startIt;
  str->length = nulIt - startIt;
  pp->pos += str->length + 1;
  return mlirBytecodeSuccess();
}

static MlirBytecodeStream
mbci_populateParserPosForSection(MlirBytecodeBytesRef ref) {
  MlirBytecodeStream pp;
  pp.start = pp.pos = ref.data;
  pp.end = pp.start + ref.length;
  return pp;
}

MlirBytecodeStream mlirBytecodeStreamCreate(MlirBytecodeBytesRef bytes) {
  MlirBytecodeStream ret = {0};
  ret.start = ret.pos = bytes.data;
  ret.end = ret.start + bytes.length;
  return ret;
}

static MlirBytecodeBytesRef mbci_getSection(const MlirBytecodeFile *const file,
                                            int index) {
  return ((MlirFileInternal *)file)->sectionData[index];
}

MlirBytecodeStatus
mlirBytecodeForEachString(void *callerState,
                          const MlirBytecodeFile *const mlirFile,
                          MlirBytecodeStringCallBack fn) {
  const MlirBytecodeBytesRef stringSection =
      mbci_getSection(mlirFile, mbci_kString);
  MlirBytecodeStream pp = mbci_populateParserPosForSection(stringSection);

  uint64_t numStrings;
  if (mlirBytecodeFailed(mbci_parseVarInt(&pp, &numStrings)))
    return mlirBytecodeEmitError("failed to parse number of strings");

  // Parse each of the strings. The sizes of the strings are encoded in reverse
  // order, so that's the order we populate the table.
  size_t stringDataEndOffset = stringSection.length;
  for (int i = numStrings; i > 0; --i) {
    uint64_t stringSize;
    if (mlirBytecodeFailed(mbci_parseVarInt(&pp, &stringSize)))
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
    if (mlirBytecodeFailed(fn(callerState,
                              (MlirBytecodeStringHandle){.id = i - 1},
                              numStrings, str)))
      return mlirBytecodeEmitError("string callback failed");
    stringDataEndOffset = stringOffset;
  }

  // Check that the only remaining data was for the strings, i.e. the reader
  // should be at the same offset as the first string.
  if (pp.pos != (pp.start + stringDataEndOffset)) {
    return mlirBytecodeEmitError(
        "unexpected trailing data between the offsets for strings "
        "and their data");
  }
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeParseDialectSection(void *callerState, MlirBytecodeFile *mlirFile,
                                MlirBytecodeDialectCallBack dialectFn,
                                MlirBytecodeDialectOpCallBack opFn) {
  const MlirBytecodeBytesRef dialectSection =
      mbci_getSection(mlirFile, mbci_kDialect);
  MlirBytecodeStream pp = mbci_populateParserPosForSection(dialectSection);
  mlirBytecodeEmitDebug("Parsing dialect section of length %ld",
                        dialectSection.length);

  uint64_t numDialects;
  if (mlirBytecodeFailed(mbci_parseVarInt(&pp, &numDialects)))
    return mlirBytecodeEmitError("unable to parse number of dialects");
  mlirBytecodeEmitDebug("number of dialects = %d", numDialects);

  uint64_t dialectName;
  for (uint64_t i = 0; i < numDialects; ++i) {
    if (mlirBytecodeFailed(mbci_parseVarInt(&pp, &dialectName)))
      return mlirBytecodeEmitError("unable to parse dialect %d", i);
    mlirBytecodeEmitDebug("dialect[%d] = %s", i,
                          mlirBytecodeGetStringSectionValue(
                              callerState, mlirFile,
                              (MlirBytecodeDialectHandle){.id = dialectName})
                              .data);
    if (mlirBytecodeFailed(dialectFn(
            callerState, (MlirBytecodeDialectHandle){.id = i}, numDialects,
            (MlirBytecodeStringHandle){.id = dialectName})))
      return mlirBytecodeFailure();
  }

  while (!mbci_streamEmpty(&pp)) {
    uint64_t dialect, numOpNames;
    mbci_parseVarInt(&pp, &dialect);
    mbci_parseVarInt(&pp, &numOpNames);

    mlirBytecodeEmitDebug("parsing for dialect %d %d ops", dialect, numOpNames);
    for (uint64_t j = 0; j < numOpNames; ++j) {
      uint64_t opName;
      if (mlirBytecodeFailed(mbci_parseVarInt(&pp, &opName)))
        return mlirBytecodeEmitError("failed to parse op name");
      mlirBytecodeEmitDebug(
          "\top[%d] = %s (%d) . %s (%d)", (int)index,
          mlirBytecodeGetStringSectionValue(
              callerState, mlirFile, (MlirBytecodeDialectHandle){.id = dialect})
              .data,
          dialect,
          mlirBytecodeGetStringSectionValue(
              callerState, mlirFile, (MlirBytecodeStringHandle){.id = opName})
              .data,
          opName);
      // Associate op[dialect][j] = opName
      if (mlirBytecodeFailed(opFn(callerState,
                                  (MlirBytecodeDialectHandle){.id = dialect},
                                  (MlirBytecodeOpHandle){.id = j},
                                  (MlirBytecodeStringHandle){.id = opName})))
        return mlirBytecodeFailure();
    }
  }

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeForEachAttributeAndType(
    void *callerState, MlirBytecodeFile *mlirFile,
    MlirBytecodeAttrCallBack attrFn, MlirBytecodeTypeCallBack typeFn) {
  const MlirBytecodeBytesRef offsetSection =
      mbci_getSection(mlirFile, mbci_kAttrTypeOffset);
  MlirBytecodeStream offsetPP = mbci_populateParserPosForSection(offsetSection);
  const MlirBytecodeBytesRef atSection =
      mbci_getSection(mlirFile, mbci_kAttrType);
  MlirBytecodeStream atPP = mbci_populateParserPosForSection(atSection);

  uint64_t numAttrs, numTypes;
  if (mlirBytecodeFailed(mbci_parseVarInt(&offsetPP, &numAttrs)) ||
      mlirBytecodeFailed(mbci_parseVarInt(&offsetPP, &numTypes)))
    return mlirBytecodeEmitError("invalid number of attributes or types");
  mlirBytecodeEmitDebug("parsing %ld attributes and %ld types", numAttrs,
                        numTypes);

  uint64_t i = 0;
  while (i < numAttrs) {
    uint64_t dialect;
    if (mlirBytecodeFailed(mbci_parseVarInt(&offsetPP, &dialect)))
      return mlirBytecodeEmitError(
          "invalid dialect handle while parsing attributes");

    uint64_t numElements;
    if (mlirBytecodeFailed(mbci_parseVarInt(&offsetPP, &numElements)))
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
      MlirBytecodeStatus ret =
          attrFn(callerState, (MlirBytecodeDialectHandle){.id = dialect},
                 (MlirBytecodeAttrHandle){.id = i++}, numAttrs,
                 hasCustomEncoding, attr);
      if (mlirBytecodeFailed(ret))
        // TODO: Should we rely that instantiations will always do this? Perhaps
        // only emit debug information here?
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
    if (mlirBytecodeFailed(mbci_parseVarInt(&offsetPP, &dialect)))
      return mlirBytecodeEmitError(
          "invalid dialect handle while parsing types");

    uint64_t numElements;
    if (mlirBytecodeFailed(mbci_parseVarInt(&offsetPP, &numElements)))
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

      MlirBytecodeStatus ret =
          typeFn(callerState, (MlirBytecodeDialectHandle){.id = dialect},
                 (MlirBytecodeTypeHandle){.id = i++}, numTypes,
                 hasCustomEncoding, type);
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
mlirBytecodeForEachResource(void *callerState, MlirBytecodeFile *const mlirFile,
                            MlirBytecodeResourceCallBack fn) {
  const MlirBytecodeBytesRef offsetSection =
      mbci_getSection(mlirFile, mbci_kResourceOffset);
  MlirBytecodeStream offsetPP = mbci_populateParserPosForSection(offsetSection);
  const MlirBytecodeBytesRef resSection =
      mbci_getSection(mlirFile, mbci_kResourceOffset);
  MlirBytecodeStream resPP = mbci_populateParserPosForSection(resSection);

  uint64_t numExternalResourceGroups;
  if (mlirBytecodeFailed(
          mbci_parseVarInt(&offsetPP, &numExternalResourceGroups)))
    return mlirBytecodeEmitError(
        "invalid/missing number of external resource groups");
  if (!numExternalResourceGroups)
    return mlirBytecodeSuccess();

  for (uint64_t i = 0; i < numExternalResourceGroups; ++i) {
    uint64_t groupKey;
    if (mlirBytecodeFailed(mbci_parseVarInt(&offsetPP, &groupKey)))
      return mlirBytecodeEmitError("invalid/missing resource group key");
    mlirBytecodeEmitDebug(
        "Group for %s / %d",
        mlirBytecodeGetStringSectionValue(
            callerState, mlirFile, (MlirBytecodeStringHandle){.id = groupKey})
            .data,
        numExternalResourceGroups);

    uint64_t numResources;
    if (mlirBytecodeFailed(mbci_parseVarInt(&offsetPP, &numResources)))
      return mlirBytecodeEmitError("invalid/missing number of resources");

    for (uint64_t j = 0; j < numResources; ++j) {
      uint64_t resourceKey;
      if (mlirBytecodeFailed(mbci_parseVarInt(&offsetPP, &resourceKey)))
        return mlirBytecodeEmitError("invalid/missing resource key");
      uint64_t size;
      if (mlirBytecodeFailed(mbci_parseVarInt(&offsetPP, &size)))
        return mlirBytecodeEmitError("invalid/missing resource size");
      uint8_t kind;
      if (mlirBytecodeFailed(mbci_parseByte(&offsetPP, &kind)))
        return mlirBytecodeEmitError("invalid/missing resource kind");

      MlirBytecodeAsmResourceEntryKind resKind = kind;
      switch (resKind) {
      case kMlirBytecodeResourceEntryBool: {
        uint8_t value;
        if (mlirBytecodeFailed(mbci_parseByte(&resPP, &value)))
          return mlirBytecodeEmitError("invalid/missing resource entry");
        if (mlirBytecodeFailed(fn(callerState,
                                  (MlirBytecodeStringHandle){.id = groupKey},
                                  numExternalResourceGroups,
                                  (MlirBytecodeStringHandle){.id = resourceKey},
                                  resKind, NULL, &value, NULL)))
          return mlirBytecodeFailure();
        break;
      }
      case kMlirBytecodeResourceEntryBlob: {
        uint64_t alignment;
        if (mlirBytecodeFailed(mbci_parseVarInt(&resPP, &alignment)))
          return mlirBytecodeEmitError("invalid/missing resource alignment");
        uint64_t size;
        if (mlirBytecodeFailed(mbci_parseVarInt(&resPP, &size)))
          return mlirBytecodeEmitError("invalid/missing resource size");
        if (mlirBytecodeFailed(mbci_alignTo(&resPP, alignment)))
          return mlirBytecodeFailure();
        MlirBytecodeBytesRef blob = {.data = resPP.pos, .length = size};
        if (mlirBytecodeFailed(fn(callerState,
                                  (MlirBytecodeStringHandle){.id = groupKey},
                                  numExternalResourceGroups,
                                  (MlirBytecodeStringHandle){.id = resourceKey},
                                  resKind, &blob, NULL, NULL)))
          return mlirBytecodeFailure();
        break;
      }
      case kMlirBytecodeResourceEntryString: {
        uint64_t value;
        if (mlirBytecodeFailed(mbci_parseVarInt(&resPP, &value)))
          return mlirBytecodeEmitError("invalid/missing resource value");
        if (mlirBytecodeFailed(
                fn(callerState, (MlirBytecodeStringHandle){.id = groupKey},
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

bool mlirBytecodeIsSentinel(MlirBytecodeAttrHandle attr) {
  return attr.id == (uint64_t)kMlirBytecodeHandleSentinel;
}

// TODO: Add option to dynamically allocate.
// Reserve top most for in-process op construction with or without regions.
const int mbci_MlirIRSectionStackMaxDepth = 16 - 1;

// We keep the following on the stack for a op with regions when descending into
// regions:
struct mbci_MlirIRSectionStackEntry {
  // State of parent op as currently populated.
  MlirBytecodeOperationState op;

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

  mbci_MlirIRSectionStackEntry data[mbci_MlirIRSectionStackMaxDepth + 1];
};
typedef struct mbci_MlirIRSectionStack mbci_MlirIRSectionStack;

static mbci_MlirIRSectionStackEntry* mbci_getStackTop(mbci_MlirIRSectionStack *stack) {
  return &stack->data[stack->top];
}
static mbci_MlirIRSectionStackEntry* mbci_getStackWIP(mbci_MlirIRSectionStack *stack) {
  return &stack->data[stack->top+1];
}

// Initialize stack entry with required state to resume. `op` is initialized
// already and so not passed in.
static MlirBytecodeStatus
mbci_mlirIRSectionStackPush(
  uint16_t numRegionsRemaining,
  uint16_t numBlocksRemaining,
  uint16_t numOpsRemaining,
                                mbci_MlirIRSectionStack *stack) {
  if (stack->top == mbci_MlirIRSectionStackMaxDepth ) {
    mlirBytecodeEmitError("IR stack max depth exceeded");
    return mlirBytecodeIterationInterrupt();
  }
                                  ++stack->top;
  mbci_MlirIRSectionStackEntry* it = mbci_getStackTop(stack);
  it->numRegionsRemaining = numRegionsRemaining;
  it->numBlocksRemaining = numBlocksRemaining;
  it->numOpsRemaining = numOpsRemaining;

  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus
mbci_mlirIRSectionStackPop(mbci_MlirIRSectionStack *stack) {
  if (stack->top == 0)
    return mlirBytecodeEmitError("IR stack exhausted");
                                  --stack->top;
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus mbci_parseBlock(MlirBytecodeStream *pp,
                                          mbci_MlirIRSectionStack *stack,
                                          void *callerState,
                                          MlirBytecodeBlockEnter blockEnterFn) {
  bool hasArgs;
  uint64_t numOps;
  if (mlirBytecodeFailed(
          mlirBytecodeParseVarIntWithFlag(pp, &numOps, &hasArgs)))
    return mlirBytecodeEmitError("unable to parse block's number of args");

  const uint8_t *start = pp->pos;
  uint64_t numArgs = 0;
  if (hasArgs) {
    if (mlirBytecodeFailed(mbci_parseVarInt(pp, &numArgs)))
      return mlirBytecodeEmitError("invalid/missing number of block args");
    start = pp->pos;
    if (mlirBytecodeFailed(mbci_skipVarInts(pp, 2 * numArgs)))
      return mlirBytecodeEmitError("invalid/missing type or location");
  }
  MlirBlockArgHandleIterator blockArgs = {.count = numArgs};
  blockArgs.stream.start = blockArgs.stream.pos = start;
  blockArgs.stream.end = pp->pos;

  mbci_mlirIRSectionStackDec(stack);
  if (mlirBytecodeFailed(blockEnterFn(callerState, &blockArgs, numOps)))
    return mlirBytecodeFailure();

  mbci_mlirIRSectionStackPushBack(kStackOperation, numOps, stack);
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus
mbci_parseRegions(MlirBytecodeStream *pp, mbci_MlirIRSectionStack *stack,
                 int numRegions,
                 void *state, 
                 MlirBytecodeRegionEnter regionEnterFn,
                 MlirBytecodeBlockEnter blockEnterFn,
                 MlirBytecodeBlockExit blockExitFn,
                 MlirBytecodeRegionExit regionExitFn,
                 bool isIsolated) {
  uint64_t numBlocks;
  if (mlirBytecodeFailed(mbci_parseVarInt(pp, &numBlocks)))
    return mlirBytecodeEmitError("invalid number of blocks");
  if (numBlocks == 0)
    return mlirBytecodeSuccess();

  uint64_t numValues;
  if (mlirBytecodeFailed(mbci_parseVarInt(pp, &numValues)))
    return mlirBytecodeEmitError("invalid/missing number of values in region");

  mbci_mlirIRSectionStackDec(stack);
  if (mlirBytecodeFailed(
          regionEnterFn(state, isIsolated, numBlocks, numValues)))
    return mlirBytecodeFailure();

  mbci_mlirIRSectionStackPushBack(
      isIsolated ? kStackBlockIsolated : kStackBlock, numBlocks, stack);
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus mbci_parseOperation(MlirBytecodeStream *pp,
                                              mbci_MlirIRSectionStack *stack,
                                              void *callerState,
                                              MlirBytecodeOperation opFn) {
  enum OpEncodingMask {
    kHasAttrs = 0x01,
    kHasResults = 0x02,
    kHasOperands = 0x04,
    kHasSuccessors = 0x08,
    kHasInlineRegions = 0x10,
  };

  uint64_t name;
  if (mlirBytecodeFailed(mbci_parseVarInt(pp, &name)))
    return mlirBytecodeEmitError("invalid operation");

  uint8_t encodingMask;
  if (mlirBytecodeFailed(mbci_parseByte(pp, &encodingMask)))
    return mlirBytecodeEmitError("invalid encoding mask");

  MlirBytecodeAttrHandle loc;
  if (mlirBytecodeFailed(mbci_parseVarInt(pp, (uint64_t *)&loc)))
    return mlirBytecodeEmitError("invalid operation location");

  mbci_MlirIRSectionStackEntry* cur = mbci_getStackWIP(stack);
  MlirBytecodeStatus ret = mlirBytecodeOperationStatePush(callerState, cur);
  if (!mlirBytecodeSucceeded(ret)) return ret;

  if (encodingMask & kHasAttrs) {
    MlirBytecodeAttrHandle attrDict;
    if (mlirBytecodeFailed(mlirBytecodeReadAttrHandle(pp, &attrDict))) {
      // TODO: add attributes here instead of using sentinel.
      return mlirBytecodeEmitError("invalid op attribute handle");
    }
    ret = mlirBytecodeOperationStateAddAttributes(callerState, cur, attrDict);
  }

  // TODO: avoid double parsing.

  // Parsing all the variadic sizes.
  // This could have been done by using additional memory instead of
  // reparsing, but keeping number of allocations to minimum.
  uint64_t numResults = 0;
  const uint8_t *start = pp->pos;
  if ((encodingMask & kHasResults)) {
    if (mlirBytecodeSucceeded(mbci_parseVarInt(pp, &numResults))) {
      start = pp->pos;
      if (mlirBytecodeFailed(mbci_skipVarInts(pp, numResults)))
        return mlirBytecodeEmitError("invalid result type");
    } else {
      return mlirBytecodeEmitError("invalid number of results");
    }
  }
  MlirBytecodeHandleIterator resultTypes = {.count = numResults};
  resultTypes.stream.start = resultTypes.stream.pos = start;
  resultTypes.stream.end = pp->pos;

  uint64_t numOperands = 0;
  start = pp->pos;
  if ((encodingMask & kHasOperands)) {
    if (mlirBytecodeSucceeded(mbci_parseVarInt(pp, &numOperands))) {
      start = pp->pos;
      if (mlirBytecodeFailed(mbci_skipVarInts(pp, numOperands)))
        return mlirBytecodeEmitError("invalid operand");
    } else {
      return mlirBytecodeEmitError("invalid number of operands");
    }
  }
  MlirBytecodeHandleIterator operands = {.count = numOperands};
  operands.stream.start = operands.stream.pos = start;
  operands.stream.end = pp->pos;

  uint64_t numSuccessors = 0;
  start = pp->pos;
  if ((encodingMask & kHasSuccessors)) {
    if (mlirBytecodeSucceeded(mbci_parseVarInt(pp, &numSuccessors))) {
      start = pp->pos;
      if (mlirBytecodeFailed(mbci_skipVarInts(pp, numSuccessors)))
        return mlirBytecodeEmitError("invalid operand");
    } else {
      return mlirBytecodeEmitError("invalid number of successors");
    }
  }
  MlirBytecodeHandleIterator successors = {.count = numSuccessors};
  successors.stream.start = successors.stream.pos = start;
  successors.stream.end = pp->end;

  uint64_t numRegions = 0;
  bool isIsolatedFromAbove = false;
  if (encodingMask & kHasInlineRegions) {
    if (mlirBytecodeFailed(mlirBytecodeParseVarIntWithFlag(
            pp, &numRegions, &isIsolatedFromAbove))) {
      return mlirBytecodeEmitError("invalid number of regions");
    }
  } else {
    ret = mlirBytecodeOperationStatePop(callerState,cur);
    return mlirBytecodeSuccess();
  }

  mbci_mlirIRSectionStackDec(stack);
  if (mlirBytecodeFailed(opFn(
          callerState, (MlirBytecodeOpHandle){.id = name}, attrDict,
          (MlirBytecodeStream *)&resultTypes, (MlirBytecodeStream *)&operands,
          (MlirBytecodeStream *)&successors, isIsolatedFromAbove, numRegions)))
    return mlirBytecodeEmitError("op function callback failed");

  if (numRegions > 0) {
    mbci_mlirIRSectionStackPushBack(isIsolatedFromAbove ? kStackRegionIsolated
                                                        : kStackRegion,
                                    numRegions, stack);
  }

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeParseIRSection(
    void *callerState, MlirBytecodeFile *mlirFile, MlirBytecodeOperation opFn,
    MlirBytecodeRegionEnter regionEnterFn, MlirBytecodeBlockEnter blockEnterFn,
    MlirBytecodeBlockExit blockExitFn, MlirBytecodeRegionExit regionExitFn,
    MlirBytecodeIsolatedOperationExit opExitFn) {
  const MlirBytecodeBytesRef irSection = mbci_getSection(mlirFile, mbci_kIR);
  MlirBytecodeStream pp = mbci_populateParserPosForSection(irSection);

  bool hasArgs;
  uint64_t numOps;
  if (mlirBytecodeFailed(
          mlirBytecodeParseVarIntWithFlag(&pp, &numOps, &hasArgs)))
    return mlirBytecodeFailure();
  if (hasArgs)
    return mlirBytecodeEmitError("IR section may not have block args");
  if (numOps != 1)
    return mlirBytecodeEmitError("only one top-level op supported");

  // P
  mbci_MlirIRSectionStack stack = {.top = -1};
  mbci_parseOperation(&pp, &stack, callerState, opFn);

  mbci_mlirIRSectionStackPushBack(kStackOperation, numOps, &stack);

  while (stack.top != -1) {
    uint64_t top = stack.data[stack.top];

    mbci_IRStackType cur = top & ((1 << mbci_irStackShift) - 1);

    if ((top >> mbci_irStackShift) == 0) {
      if (cur == kStackOperation &&
          mlirBytecodeFailed(blockExitFn(callerState)))
        return mlirBytecodeFailure();
      if ((cur == kStackBlock || cur == kStackBlockIsolated) &&
          mlirBytecodeFailed(
              regionExitFn(callerState, cur == kStackBlockIsolated)))
        return mlirBytecodeFailure();
      if ((cur == kStackRegion || cur == kStackRegionIsolated) &&
          mlirBytecodeFailed(
              opExitFn(callerState, cur == kStackRegionIsolated)))
        return mlirBytecodeFailure();
      --stack.top;
      continue;
    }

    switch (cur) {
    case kStackBlock:
    case kStackBlockIsolated:
      if (mlirBytecodeFailed(
              mbci_parseBlock(&pp, &stack, callerState, blockEnterFn)))
        return mlirBytecodeFailure();
      break;
    case kStackOperation:
      if (mlirBytecodeFailed(
              mbci_parseOperation(&pp, &stack, callerState, opFn)))
        return mlirBytecodeFailure();
      break;
    case kStackRegionIsolated:
    case kStackRegion:
      if (mlirBytecodeFailed(mbci_parseRegion(&pp, &stack,
      callerState,
                                              regionEnterFn,
                                              blockEnterFn,
                                              blockExitFn,
                                              regionExitFn,
                                              cur == kStackRegionIsolated)))
        return mlirBytecodeFailure();
      break;
    }
  }
  return mlirBytecodeSuccess();
}

bool mlirBytecodeFileEmpty(MlirBytecodeFile *file) {
  return ((MlirFileInternal *)file->b)->producer.length == 0;
}

MlirBytecodeFile mlirBytecodePopulateFile(const MlirBytecodeBytesRef bytes) {
  MlirBytecodeFile ret = {.b = 0};
  MlirFileInternal *mlirFile = (MlirFileInternal *)&ret.b;
  MlirBytecodeStream pp = mbci_populateParserPosForSection(bytes);

  uint8_t magic[4];
  if (mlirBytecodeFailed(mbci_parseBytes(&pp, magic, 4)))
    return mlirBytecodeEmitError("unable to read 4 byte magic code"), ret;
  if (magic[0] != 'M' || magic[1] != 'L' || magic[2] != 0xef || magic[3] != 'R')
    return mlirBytecodeEmitError("invalid file magic code"), ret;

  // Parse the bytecode version and producer.
  MlirBytecodeBytesRef producer;
  if (mlirBytecodeFailed(mbci_parseVarInt(&pp, &mlirFile->version)) ||
      mlirBytecodeFailed(mlirBytecodeParseNullTerminatedString(&pp, &producer)))
    return mlirBytecodeEmitError("invalid version or producer"), ret;
  mlirBytecodeEmitDebug("Producer: %s\n", producer.data);

  while (!mbci_streamEmpty(&pp)) {
    // Read the next section from the bytecode.
    if (mlirBytecodeFailed(mbci_parseSections(&pp, mlirFile->sectionData)))
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
  mlirFile->producer = producer;
  return ret;
}

MlirBytecodeStatus mlirBytecodeParseFile(
    void *callerState, MlirBytecodeBytesRef bytes,
    MlirBytecodeAttrCallBack attrFn, MlirBytecodeTypeCallBack typeFn,
    MlirBytecodeDialectCallBack dialectFn,
    MlirBytecodeDialectOpCallBack dialectOpFn,
    MlirBytecodeResourceCallBack resourceFn,
    MlirBytecodeStringCallBack stringFn, MlirBytecodeOperation opFn,
    MlirBytecodeRegionEnter regionEnterFn, MlirBytecodeBlockEnter blockEnterFn,
    MlirBytecodeBlockExit blockExitFn, MlirBytecodeRegionExit regionExitFn,
    MlirBytecodeIsolatedOperationExit opExitFn) {
  static_assert(sizeof(MlirFileInternal) == sizeof(MlirBytecodeFile),
                "MlirBytecodeFile shell type size mismatch");
  static_assert(alignof(MlirFileInternal) == alignof(MlirBytecodeFile),
                "MlirBytecodeFile shell type alignment mismatch");

  MlirBytecodeFile mlirFile = mlirBytecodePopulateFile(bytes);
  if (mlirBytecodeFileEmpty(&mlirFile))
    return mlirBytecodeFailure();

  // Process the string section first.
  // Finally, process the IR section.
  if (mlirBytecodeFailed(
          mlirBytecodeForEachString(callerState, &mlirFile, stringFn)) ||
      // Process the dialect section.
      mlirBytecodeFailed(mlirBytecodeParseDialectSection(
          callerState, &mlirFile, dialectFn, dialectOpFn)) ||
      // Process the resource section if present.
      mlirBytecodeFailed(
          mlirBytecodeForEachResource(callerState, &mlirFile, resourceFn)) ||
      // Process the attribute and type section.
      mlirBytecodeFailed(mlirBytecodeForEachAttributeAndType(
          callerState, &mlirFile, attrFn, typeFn)) ||
      // Finally, process the IR section.
      mlirBytecodeFailed(mlirBytecodeParseIRSection(
          callerState, &mlirFile, opFn, regionEnterFn, blockEnterFn,
          blockExitFn, regionExitFn, opExitFn)))
    return mlirBytecodeFailure();

  return mlirBytecodeSuccess();
}

// ---

// Helper struct & function for iterating until specific value.
struct mbci_IterateStruct {
  MlirBytecodeBytesRef *ret;
  size_t n;
};

MlirBytecodeStatus mbci_iterateUntilN(void *state, MlirBytecodeStringHandle i,
                                      size_t total, MlirBytecodeBytesRef ref) {
  (void)total;
  struct mbci_IterateStruct *s = (struct mbci_IterateStruct *)state;
  if (i.id == s->n) {
    *s->ret = ref;
    return mlirBytecodeIterationInterrupt();
  }
  return mlirBytecodeSuccess();
}

__attribute__((weak)) MlirBytecodeBytesRef
mlirBytecodeGetStringSectionValue(void *callerState,
                                  const MlirBytecodeFile *mlirFile,
                                  MlirBytecodeStringHandle idx) {
  MlirBytecodeBytesRef ret = {.data = 0, .length = 0};
  struct mbci_IterateStruct state = {.ret = &ret, .n = idx.id};
  MlirBytecodeStatus status =
      mlirBytecodeForEachString(&state, mlirFile, mbci_iterateUntilN);
  assert(mlirBytecodeSucceeded(status));
  return ret;
}

__attribute__((weak)) MlirBytecodeOpRef
mlirBytecodeGetOpName(void *callerState, const MlirBytecodeFile *mlirFile,
                      MlirBytecodeOpHandle hdl) {
  const MlirBytecodeBytesRef dialectSection =
      mbci_getSection(mlirFile, mbci_kDialect);
  MlirBytecodeStream pp = mbci_populateParserPosForSection(dialectSection);

  MlirBytecodeOpRef ret = {.dialect = {-1}, .op = {-1}};
  uint64_t numDialects;
  if (mlirBytecodeFailed(mbci_parseVarInt(&pp, &numDialects)))
    return mlirBytecodeEmitError("unable to parse number of dialects"), ret;

  if (mlirBytecodeFailed(mbci_skipVarInts(&pp, numDialects)))
    return ret;

  uint64_t index = 0;
  while (!mbci_streamEmpty(&pp)) {
    uint64_t dialect, numOpNames;
    if (mlirBytecodeFailed(mbci_parseVarInt(&pp, &dialect)) ||
        mlirBytecodeFailed(mbci_parseVarInt(&pp, &numOpNames)))
      return ret;

    for (uint64_t j = 0; j < numOpNames; ++j) {
      uint64_t opName;
      if (mlirBytecodeFailed(mbci_parseVarInt(&pp, &opName)))
        return mlirBytecodeEmitError("invalid op name"), ret;
      if (index == (uint64_t)hdl.id) {
        ret.dialect.id = dialect;
        ret.op.id = opName;
        return ret;
      }
      ++index;
    }
  }

  mlirBytecodeEmitError("unable to find references string %ld", hdl);
  return ret;
}

#endif // MLIRBC_PARSE_IMPLEMENTATION
