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
// parser. It is defined as a pure header-based implementation. Set
//   #define MLIRBC_PARSE_IMPLEMENTATION
// before you include this file in one C or C++ file to create the
// implementation.

// This file should be included before all dialect parsing extensions.

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

#define MLIRBC_DEF extern

#ifdef __cplusplus
extern "C" {
#endif

// Populates the MlirBytecodeFile contents for given in memory bytes.
// Returns an empty file if population failed.
MlirBytecodeFile mlirBytecodePopulateFile(MlirBytecodeBytesRef bytes);

//===----------------------------------------------------------------------===//
// Callbacks invoked by the section parsing functions below.

// Callbacks accept an opaque pointer as first argument that gets directly
// propagated during parsing and can be used by instantiator for
// additional/parse local state capture.

// Associate dialect with string handle.
typedef MlirBytecodeStatus (*MlirBytecodeDialectCallBack)(
    void *, MlirBytecodeDialectHandle, size_t /*total*/,
    MlirBytecodeStringHandle);

// Associate dialect and opname with string handle.
typedef MlirBytecodeStatus (*MlirBytecodeDialectOpCallBack)(
    void *, MlirBytecodeDialectHandle, MlirBytecodeOpHandle, size_t /*total*/,
    MlirBytecodeStringHandle);

// Associate dialect attribute with range in memory.
typedef MlirBytecodeStatus (*MlirBytecodeAttrCallBack)(
    void *, MlirBytecodeDialectHandle, MlirBytecodeAttrHandle, size_t /*total*/,
    bool, MlirBytecodeBytesRef);

// Associate dialect type with range in memory.
typedef MlirBytecodeStatus (*MlirBytecodeTypeCallBack)(
    void *, MlirBytecodeDialectHandle, MlirBytecodeTypeHandle, size_t /*total*/,
    bool, MlirBytecodeBytesRef);

// Associate groupKey[resourceKey] with either MlirBytecodeStatus, string or
// blob.
typedef MlirBytecodeStatus (*MlirBytecodeResourceCallBack)(
    void *, MlirBytecodeStringHandle groupKey, MlirBytecodeSize totalGroups,
    MlirBytecodeStringHandle resourceKey, MlirBytecodeAsmResourceEntryKind,
    const MlirBytecodeBytesRef *blob, const uint8_t *MlirBytecodeStatusResource,
    const MlirBytecodeStringHandle *);

// String callback which consists of string handle, total number of strings in
// string section and bytes corresponding to the string.
typedef MlirBytecodeStatus (*MlirBytecodeStringCallBack)(
    void *, MlirBytecodeStringHandle, size_t /*total*/, MlirBytecodeBytesRef);

// IR Parsing handlers.

// Called when entering block in region. The blockArgs consists of pair
// type and location and numOps is the number of ops in the block.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeBlockEnter(
    void *, MlirBlockArgHandleIterator *blockArgs, size_t numOps);
// Called when exiting the block.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeBlockExit(void *);

// Called per operation in block.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperation(
    void *, MlirBytecodeOpHandle name, MlirBytecodeAttrHandle attrDict,
    MlirBytecodeStream *resultTypes, MlirBytecodeStream *operands,
    MlirBytecodeStream *succcessors, bool isIsolatedFromAbove,
    size_t numRegions);

// Called post completed parsing of an isolated from above operation.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeIsolatedOperationExit(void *);

// Called when entering a region with numBlocks blocks and numValues Values
// (including values due to block args).
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeRegionEnter(void *, bool isIsolated,
                                                      size_t numBlocks,
                                                      size_t numValues);

// Called when entering a region.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeRegionExit(void *, bool isIsolated);

// Parses the given MLIR file represented in memory `bytes`, calls the
// appropriate callbacks during parsing. This combines the parsing methods
// below.
MlirBytecodeStatus parseMlirFile(MlirBytecodeBytesRef bytes, void *callerState,
                                 MlirBytecodeAttrCallBack attrFn,
                                 MlirBytecodeTypeCallBack typeFn,
                                 MlirBytecodeDialectCallBack dialectFn,
                                 MlirBytecodeDialectOpCallBack dialectOpFn,
                                 MlirBytecodeResourceCallBack resourceFn,
                                 MlirBytecodeStringCallBack stringFn);

// Iterators over attributes and types, calling MlirBytecodeAttrCallBack and
// MlirBytecodeTypeCallBack upon encountering Attribute or Type respectively.
// Returns whether failed.
MlirBytecodeStatus mlirBytecodeForEachAttributeAndType(
    MlirBytecodeFile *mlirFile, void *callerState,
    MlirBytecodeAttrCallBack attrFn, MlirBytecodeTypeCallBack typeFn);

// Parses the dialect section, invoking MlirBytecodeDialectCallBack upon dialect
// enountered and MlirBytecodeDialectOpCallBack per operation type in dialect.
// Returns whether failed.
MlirBytecodeStatus
mlirBytecodeParseDialectSection(MlirBytecodeFile *mlirFile, void *callerState,
                                MlirBytecodeDialectCallBack dialectFn,
                                MlirBytecodeDialectOpCallBack opFn);

// Parse IR section in mlirFile. The block args, operation and region callback
// are invoked during bytecode in-order walk. Additionally allows for passing in
// an opaque state.
//
// The IR section parsing follows the nesting order:
//   op ->* regions ->* blocks
// The caller is required to keep track of when all operations/blocks in
// block/region have been processed and so parsing resumes at parent level.
// Returns whether failed.
MlirBytecodeStatus mlirBytecodeParseIRSection(MlirBytecodeFile *mlirFile,
                                              void *callerState);

// Parse the resource section, calling MlirBytecodeResourceCallBack upon
// resources encountered. Returns whether failed.
MlirBytecodeStatus
mlirBytecodeParseResourceSection(MlirBytecodeFile *mlirFile, void *callerState,
                                 MlirBytecodeResourceCallBack fn);

// Nnvoke the callback per string in string section.
// Returns whether failed.
MlirBytecodeStatus
mlirBytecodeForEachString(void *callerState,
                          const MlirBytecodeFile *const mlirFile,
                          MlirBytecodeStringCallBack fn);

// Returns whether the given MlirBytecodeFile structure is empty.
bool mlirBytecodeFileEmpty(MlirBytecodeFile *file);

//===----------------------------------------------------------------------===//
// Lazy parsing methods

// Returns the requested string from the string section.
// Note: this doesn't cache any state.
MlirBytecodeBytesRef
mlirBytecodeGetStringSectionValue(const MlirBytecodeFile *const mlirFile,
                                  MlirBytecodeStringHandle idx);

// Returns the requested dialect & operation for given index.
// Note: this doesn't cache any state.
MlirBytecodeOpRef
mlirBytecodeGetInstructionString(const MlirBytecodeFile *const mlirFile,
                                 MlirBytecodeOpHandle hdl);

//===----------------------------------------------------------------------===//
// Dialect parsing utility methods.
// Note: These should probably go into their own header.

// Creates a bytecode stream from section of bytes.
MlirBytecodeStream mlirBytecodeStreamCreate(MlirBytecodeBytesRef ref);

// Reset the stream to its head.
void mlirBytecodeStreamReset(MlirBytecodeStream *iterator);

// Decode the next handle on the stream and increment stream.
MlirBytecodeStatus mlirBytecodeGetNextHandle(MlirBytecodeStream *iterator,
                                             MlirBytecodeHandle *result);

// Decode uint64 on the stream and increment stream.
MlirBytecodeStatus mlirBytecodeReadVarInt(MlirBytecodeStream *iterator,
                                          uint64_t *result);

// Decode int64 on the stream and increment stream.
MlirBytecodeStatus mlirBytecodeReadSignedVarInt(MlirBytecodeStream *iterator,
                                                int64_t *result);

// Populate the next block arg type & location and increment the iterator.
// Returns whether there was an element.
MlirBytecodeStatus
mlirBytecodeGetNextBlockArgHandles(MlirBlockArgHandleIterator *iterator,
                                   MlirBytecodeTypeHandle *,
                                   MlirBytecodeAttrHandle *);

#ifdef __cplusplus
}
#endif
#endif // MLIRBC_PARSE_H

// #ifdef MLIRBC_PARSE_IMPLEMENTATION
#if 1
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

// ----
// Bytecode constants.
// ----

enum {
  /// The current bytecode version.
  kVersion = 0,

  /// An arbitrary value used to fill alignment padding.
  kAlignmentByte = 0xCB,
};

enum ID {
  /// This section contains strings referenced within the bytecode.
  kString = 0,

  /// This section contains the dialects referenced within an IR module.
  kDialect = 1,

  /// This section contains the attributes and types referenced within an IR
  /// module.
  kAttrType = 2,

  /// This section contains the offsets for the attribute and types within the
  /// AttrType section.
  kAttrTypeOffset = 3,

  /// This section contains the list of operations serialized into the bytecode,
  /// and their nested regions/operations.
  kIR = 4,

  /// This section contains the resources of the bytecode.
  kResource = 5,

  /// This section contains the offsets of resources within the Resource
  /// section.
  kResourceOffset = 6,

  /// The total number of section types.
  kNumSections = 7,
};

enum OpEncodingMask {
  kHasAttrs = 0x01,
  kHasResults = 0x02,
  kHasOperands = 0x04,
  kHasSuccessors = 0x08,
  kHasInlineRegions = 0x10,
};

struct MlirFileInternal {
  uint64_t version;
  MlirBytecodeBytesRef producer;

  MlirBytecodeBytesRef sectionData[kNumSections];
};
typedef struct MlirFileInternal MlirFileInternal;

const char *sectionIDToString(uint8_t id) {
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

static bool isSectionOptional(int id) {
  switch (id) {
  case kResource:
  case kResourceOffset:
    return true;
  case kString:
  case kDialect:
  case kAttrType:
  case kAttrTypeOffset:
  case kIR:
  default:
    return false;
  }
}

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
__attribute__((weak)) void mlirBytecodeEmitDebugImpl(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");
}
#else
__attribute__((weak)) void mlirBytecodeEmitDebugImpl(const char *fmt, ...) {
  (void)fmt;
}
#endif

//-----
// Base parsing primites.
//=====

// Represents simple parser state.
static bool empty(MlirBytecodeStream *pp) { return pp->pos >= pp->end; }

static MlirBytecodeStatus parseByte(MlirBytecodeStream *pp, uint8_t *val) {
  if (empty(pp))
    return mlirBytecodeFailure();
  *val = *pp->pos++;
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus parseBytes(MlirBytecodeStream *pp, uint8_t *val,
                                     size_t n) {
  for (uint64_t i = 0; i < n; ++i)
    if (mlirBytecodeFailed(parseByte(pp, val++)))
      return mlirBytecodeFailure();
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus skipBytes(MlirBytecodeStream *pp, size_t n) {
  pp->pos += n;
  if (pp->pos > pp->end)
    return mlirBytecodeFailure();
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus parseVarInt(MlirBytecodeStream *pp,
                                      uint64_t *result) {
  // Compute the number of bytes needed to encode the value. Each byte can hold
  // up to 7-bits of data. We only check up to the number of bits we can encode
  // in the first byte (8).
  uint8_t head;
  if (mlirBytecodeFailed(parseByte(pp, &head)))
    return mlirBytecodeFailure();
  *result = head;
  if ((*result & 1)) {
    *result >>= 1;
    return mlirBytecodeSuccess();
  }

  if (*result == 0) {
    uint8_t *ptr = (uint8_t *)(result);
    if (mlirBytecodeFailed(parseBytes(pp, ptr, sizeof(*result))))
      return mlirBytecodeFailure();
    return mlirBytecodeSuccess();
  }

  // Parse in the remaining bytes of the value.
  uint32_t numBytes = __builtin_ctz(*result);
  uint8_t *ptr = (uint8_t *)(result) + 1;
  if (mlirBytecodeFailed(parseBytes(pp, ptr, numBytes)))
    return mlirBytecodeFailure();

  // Shift out the low-order bits that were used to mark how the value was
  // encoded.
  *result >>= (numBytes + 1);
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus skipVarInts(MlirBytecodeStream *pp, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    uint8_t head;
    if (mlirBytecodeFailed(parseByte(pp, &head)))
      return mlirBytecodeFailure();
    int numBytes = (head == 0) ? 8 : __builtin_ctz(head);
    if (mlirBytecodeFailed(skipBytes(pp, numBytes)))
      return mlirBytecodeFailure();
  }
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeGetNextHandle(MlirBytecodeStream *iterator,
                                             MlirBytecodeHandle *result) {
  return mlirBytecodeReadVarInt(iterator, &result->id);
}

void mlirBytecodeStreamReset(MlirBytecodeStream *iterator) {
  iterator->pos = iterator->start;
}

MlirBytecodeStatus mlirBytecodeReadVarInt(MlirBytecodeStream *iterator,
                                          uint64_t *result) {
  uint64_t ret;
  if (mlirBytecodeFailed(parseVarInt(iterator, &ret)))
    return mlirBytecodeFailure();
  *result = ret;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeGetNextBlockArgHandles(MlirBlockArgHandleIterator *iterator,
                                   MlirBytecodeTypeHandle *type,
                                   MlirBytecodeAttrHandle *loc) {
  if (empty((MlirBytecodeStream *)iterator))
    return mlirBytecodeFailure();
  if (mlirBytecodeFailed(
          parseVarInt((MlirBytecodeStream *)iterator, (uint64_t *)type)) ||
      mlirBytecodeFailed(
          parseVarInt((MlirBytecodeStream *)iterator, (uint64_t *)loc)))
    return mlirBytecodeFailure();
  return mlirBytecodeSuccess();
}

/// Parse a variable length encoded integer whose low bit is used to encode an
/// unrelated flag, i.e: `(integerValue << 1) | (flag ? 1 : 0)`.
static MlirBytecodeStatus parseVarIntWithFlag(MlirBytecodeStream *pp,
                                              uint64_t *result, bool *flag) {
  if (mlirBytecodeFailed(parseVarInt(pp, result)))
    return mlirBytecodeFailure();
  *flag = *result & 1;
  *result >>= 1;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeReadSignedVarInt(MlirBytecodeStream *pp,
                                                int64_t *result) {
  if (mlirBytecodeFailed(parseVarInt(pp, (uint64_t *)result)))
    return mlirBytecodeFailure();
  *result = (*result >> 1) ^ (~(*result & 1) + 1);
  return mlirBytecodeSuccess();
}

static bool isPowerOf2(uint32_t value) {
  return (value != 0) && ((value & (value - 1)) == 0);
}

// Align the current reader position to the specified alignment.
static MlirBytecodeStatus alignTo(MlirBytecodeStream *pp, uint32_t alignment) {
  if (!isPowerOf2(alignment))
    return mlirBytecodeEmitError("expected alignment to be a power-of-two");

  // Shift the reader position to the next alignment boundary.
  while ((uintptr_t)pp->pos & ((uintptr_t)alignment - 1)) {
    uint8_t padding;
    if (mlirBytecodeFailed(parseByte(pp, &padding)))
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
parseSections(MlirBytecodeStream *pp,
              MlirBytecodeBytesRef sectionData[kNumSections]) {
  uint8_t byte;
  uint64_t length;
  if (mlirBytecodeFailed(parseByte(pp, &byte)) ||
      mlirBytecodeFailed(parseVarInt(pp, &length)))
    return mlirBytecodeFailure();
  uint8_t sectionID = byte & 0x7f;

  mlirBytecodeEmitDebug("Parsing %d of %lld", sectionID, length);
  if (sectionID >= kNumSections)
    return mlirBytecodeEmitError("invalid section ID: %d", sectionID);
  if (sectionData[sectionID].data != NULL) {
    return mlirBytecodeEmitError("duplicate top-level section: %s",
                                 sectionIDToString(sectionID));
  }

  bool isAligned = byte >> 7;
  if (isAligned) {
    uint64_t alignment;
    if (mlirBytecodeFailed(parseVarInt(pp, &alignment)) ||
        mlirBytecodeFailed(alignTo(pp, alignment)))
      return mlirBytecodeFailure();
  }

  sectionData[sectionID].data = pp->pos;
  sectionData[sectionID].length = length;
  return skipBytes(pp, length);
}

static MlirBytecodeStatus parseNullTerminatedString(MlirBytecodeStream *pp,
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
populateParserPosForSection(MlirBytecodeBytesRef ref) {
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

static MlirBytecodeBytesRef getSection(const MlirBytecodeFile *const file,
                                       int index) {
  return ((MlirFileInternal *)file)->sectionData[index];
}

MlirBytecodeStatus
mlirBytecodeForEachString(void *callerState,
                          const MlirBytecodeFile *const mlirFile,
                          MlirBytecodeStringCallBack fn) {
  const MlirBytecodeBytesRef stringSection = getSection(mlirFile, kString);
  MlirBytecodeStream pp = populateParserPosForSection(stringSection);

  uint64_t numStrings;
  if (mlirBytecodeFailed(parseVarInt(&pp, &numStrings)))
    return mlirBytecodeEmitError("failed to parse number of strings");

  // Parse each of the strings. The sizes of the strings are encoded in reverse
  // order, so that's the order we populate the table.
  size_t stringDataEndOffset = stringSection.length;
  for (int i = numStrings; i > 0; --i) {
    uint64_t stringSize;
    if (mlirBytecodeFailed(parseVarInt(&pp, &stringSize)))
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
mlirBytecodeParseDialectSection(MlirBytecodeFile *mlirFile, void *callerState,
                                MlirBytecodeDialectCallBack dialectFn,
                                MlirBytecodeDialectOpCallBack opFn) {
  const MlirBytecodeBytesRef dialectSection = getSection(mlirFile, kDialect);
  MlirBytecodeStream pp = populateParserPosForSection(dialectSection);
  mlirBytecodeEmitDebug("Parsing dialect section of length %ld",
                        dialectSection.length);

  uint64_t numDialects;
  if (mlirBytecodeFailed(parseVarInt(&pp, &numDialects)))
    return mlirBytecodeEmitError("unable to parse number of dialects");
  mlirBytecodeEmitDebug("number of dialects = %d", numDialects);

  uint64_t dialectName;
  for (uint64_t i = 0; i < numDialects; ++i) {
    if (mlirBytecodeFailed(parseVarInt(&pp, &dialectName)))
      return mlirBytecodeEmitError("unable to parse dialect %d", i);
    mlirBytecodeEmitDebug(
        "dialect[%d] = %s", i,
        mlirBytecodeGetStringSectionValue(mlirFile, dialectName).data);
    if (mlirBytecodeFailed(dialectFn(
            callerState, (MlirBytecodeDialectHandle){.id = i}, numDialects,
            (MlirBytecodeStringHandle){.id = dialectName})))
      return mlirBytecodeFailure();
  }

  // Compute the total number of instructions.
  // This could have been done by using additional memory instead of reparsing,
  // but keeping number of allocations to minimum.
  // TODO: Another alternative is an iterator - that is still parsing 2x though.
  // The reason I want totalOps is to enable doing a single alloc rather than
  // needing a structure that grows.
  const uint8_t *pos = pp.pos;
  uint64_t totalOps = 0;
  while (!empty(&pp)) {
    uint64_t t;
    if (mlirBytecodeFailed(parseVarInt(&pp, &t)))
      return mlirBytecodeEmitError("failed to parse dialect in op_name_group");
    if (mlirBytecodeFailed(parseVarInt(&pp, &t)))
      return mlirBytecodeEmitError(
          "failed to parse dialect in number of ops in dialect");
    if (mlirBytecodeFailed(skipVarInts(&pp, t)))
      return mlirBytecodeEmitError("failed to parse op name");
    totalOps += t;
  }
  pp.pos = pos;

  uint64_t index = 0;
  while (!empty(&pp)) {
    uint64_t dialect, numOpNames;
    parseVarInt(&pp, &dialect);
    parseVarInt(&pp, &numOpNames);

    mlirBytecodeEmitDebug("parsing for dialect %d %d ops", dialect, numOpNames);
    for (uint64_t j = 0; j < numOpNames; ++j) {
      uint64_t opName;
      if (mlirBytecodeFailed(parseVarInt(&pp, &opName)))
        return mlirBytecodeEmitError("failed to parse op name");
      mlirBytecodeEmitDebug(
          "\top[%d/%d] = %s (%d) . %s (%d)", (int)index, (int)totalOps,
          mlirBytecodeGetStringSectionValue(mlirFile, dialect).data, dialect,
          mlirBytecodeGetStringSectionValue(mlirFile, opName).data, opName);
      // Associate dialect[index] = opName
      if (mlirBytecodeFailed(
              opFn(callerState, (MlirBytecodeDialectHandle){.id = dialect},
                   (MlirBytecodeOpHandle){.id = index++}, totalOps,
                   (MlirBytecodeStringHandle){.id = opName})))
        return mlirBytecodeFailure();
    }
  }

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeForEachAttributeAndType(
    MlirBytecodeFile *mlirFile, void *callerState,
    MlirBytecodeAttrCallBack attrFn, MlirBytecodeTypeCallBack typeFn) {
  const MlirBytecodeBytesRef offsetSection =
      getSection(mlirFile, kAttrTypeOffset);
  MlirBytecodeStream offsetPP = populateParserPosForSection(offsetSection);
  const MlirBytecodeBytesRef atSection = getSection(mlirFile, kAttrType);
  MlirBytecodeStream atPP = populateParserPosForSection(atSection);

  uint64_t numAttrs, numTypes;
  if (mlirBytecodeFailed(parseVarInt(&offsetPP, &numAttrs)) ||
      mlirBytecodeFailed(parseVarInt(&offsetPP, &numTypes)))
    return mlirBytecodeEmitError("invalid number of attributes or types");
  mlirBytecodeEmitDebug("parsing %ld attributes and %ld types", numAttrs,
                        numTypes);

  uint64_t i = 0;
  while (i < numAttrs) {
    uint64_t dialect;
    if (mlirBytecodeFailed(parseVarInt(&offsetPP, &dialect)))
      return mlirBytecodeEmitError(
          "invalid dialect handle while parsing attributes");

    uint64_t numElements;
    if (mlirBytecodeFailed(parseVarInt(&offsetPP, &numElements)))
      return mlirBytecodeEmitError(
          "invalid number of elements in attr offset group");

    for (uint64_t j = 0; j < numElements; ++j) {
      uint64_t length;
      bool hasCustomEncoding;
      if (mlirBytecodeFailed(
              parseVarIntWithFlag(&offsetPP, &length, &hasCustomEncoding)))
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
        // TODO: Should we rely that instantiators will always do this? Perhaps
        // only emit debug information here?
        return mlirBytecodeEmitError("attr callback failed");
      if (mlirBytecodeInterupted(ret))
        break;
      // Unhandled attributes are not considered error.

      if (mlirBytecodeFailed(skipBytes(&atPP, length)))
        return mlirBytecodeEmitError("invalid attr offset");
    }
  }

  i = 0;
  while (i < numTypes) {
    uint64_t dialect;
    if (mlirBytecodeFailed(parseVarInt(&offsetPP, &dialect)))
      return mlirBytecodeEmitError(
          "invalid dialect handle while parsing types");

    uint64_t numElements;
    if (mlirBytecodeFailed(parseVarInt(&offsetPP, &numElements)))
      return mlirBytecodeEmitError(
          "invalid number of elements in type offset group");

    for (uint64_t j = 0; j < numElements; ++j) {
      uint64_t offset;
      bool hasCustomEncoding;
      if (mlirBytecodeFailed(
              parseVarIntWithFlag(&offsetPP, &offset, &hasCustomEncoding)))
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
      if (mlirBytecodeInterupted(ret))
        break;
      // Unhandled attributes are not considered error.

      if (mlirBytecodeFailed(skipBytes(&atPP, offset)))
        return mlirBytecodeEmitError("invalid type offset");
    }
  }

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeParseResourceSection(MlirBytecodeFile *const mlirFile,
                                 void *callerState,
                                 MlirBytecodeResourceCallBack fn) {
  const MlirBytecodeBytesRef offsetSection =
      getSection(mlirFile, kResourceOffset);
  MlirBytecodeStream offsetPP = populateParserPosForSection(offsetSection);
  const MlirBytecodeBytesRef resSection = getSection(mlirFile, kResourceOffset);
  MlirBytecodeStream resPP = populateParserPosForSection(resSection);

  uint64_t numExternalResourceGroups;
  if (mlirBytecodeFailed(parseVarInt(&offsetPP, &numExternalResourceGroups)))
    return mlirBytecodeEmitError(
        "invalid/missing number of external resource groups");
  if (!numExternalResourceGroups)
    return mlirBytecodeSuccess();

  for (uint64_t i = 0; i < numExternalResourceGroups; ++i) {
    uint64_t groupKey;
    if (mlirBytecodeFailed(parseVarInt(&offsetPP, &groupKey)))
      return mlirBytecodeEmitError("invalid/missing resource group key");
    mlirBytecodeEmitDebug(
        "Group for %s / %d",
        mlirBytecodeGetStringSectionValue(mlirFile, groupKey).data,
        numExternalResourceGroups);

    uint64_t numResources;
    if (mlirBytecodeFailed(parseVarInt(&offsetPP, &numResources)))
      return mlirBytecodeEmitError("invalid/missing number of resources");

    for (uint64_t j = 0; j < numResources; ++j) {
      uint64_t resourceKey;
      if (mlirBytecodeFailed(parseVarInt(&offsetPP, &resourceKey)))
        return mlirBytecodeEmitError("invalid/missing resource key");
      uint64_t size;
      if (mlirBytecodeFailed(parseVarInt(&offsetPP, &size)))
        return mlirBytecodeEmitError("invalid/missing resource size");
      uint8_t kind;
      if (mlirBytecodeFailed(parseByte(&offsetPP, &kind)))
        return mlirBytecodeEmitError("invalid/missing resource kind");

      MlirBytecodeAsmResourceEntryKind resKind = kind;
      switch (resKind) {
      case kMlirBytecodeResourceEntryBool: {
        uint8_t value;
        if (mlirBytecodeFailed(parseByte(&resPP, &value)))
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
        if (mlirBytecodeFailed(parseVarInt(&resPP, &alignment)))
          return mlirBytecodeEmitError("invalid/missing resource alignment");
        uint64_t size;
        if (mlirBytecodeFailed(parseVarInt(&resPP, &size)))
          return mlirBytecodeEmitError("invalid/missing resource size");
        if (mlirBytecodeFailed(alignTo(&resPP, alignment)))
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
        if (mlirBytecodeFailed(parseVarInt(&resPP, &value)))
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

struct MlirIRSectionStack {
  int top;
  const int maxDepth;

  // TODO: This can probably be 32 bits or encoded as VarInt.
  uint64_t data[16];
};
typedef struct MlirIRSectionStack MlirIRSectionStack;
typedef enum {
  kStackOperation,
  kStackBlock,
  kStackBlockIsolated,
  kStackRegion,
  kStackRegionIsolated
} IRStackType;
const int irStackShift = 3; // Number of bits for type IRStackType.

MlirBytecodeStatus mlirIRSectionStackPushBack(IRStackType t, int counter,
                                              MlirIRSectionStack *stack) {
  if ((((uint64_t)counter << irStackShift) >> irStackShift) !=
      (uint64_t)counter)
    return mlirBytecodeEmitError("count exceeds stack encoding limit");

  uint64_t val = (counter << irStackShift) | (int)t;
  if (stack->top + 1 == stack->maxDepth)
    return mlirBytecodeEmitError("IR stack max depth exceeded");
  stack->data[++stack->top] = val;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirIRSectionStackDec(MlirIRSectionStack *stack) {
  uint64_t top = stack->data[stack->top];
  uint64_t type = top & ((1 << irStackShift) - 1);
  uint64_t count = top >> irStackShift;
  stack->data[stack->top] = ((count - 1) << irStackShift) | type;
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus parseBlock(MlirBytecodeStream *pp,
                                     MlirIRSectionStack *stack,
                                     void *callerState) {
  bool hasArgs;
  uint64_t numOps;
  if (mlirBytecodeFailed(parseVarIntWithFlag(pp, &numOps, &hasArgs)))
    return mlirBytecodeEmitError("unable to parse block's number of args");

  const uint8_t *start = pp->pos;
  uint64_t numArgs = 0;
  if (hasArgs) {
    if (mlirBytecodeFailed(parseVarInt(pp, &numArgs)))
      return mlirBytecodeEmitError("invalid/missing number of block args");
    start = pp->pos;
    if (mlirBytecodeFailed(skipVarInts(pp, 2 * numArgs)))
      return mlirBytecodeEmitError("invalid/missing type or location");
  }
  MlirBlockArgHandleIterator blockArgs = {.count = numArgs};
  blockArgs.stream.start = blockArgs.stream.pos = start;
  blockArgs.stream.end = pp->pos;

  mlirIRSectionStackDec(stack);
  if (mlirBytecodeFailed(
          mlirBytecodeBlockEnter(callerState, &blockArgs, numOps)))
    return mlirBytecodeFailure();

  mlirIRSectionStackPushBack(kStackOperation, numOps, stack);
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus parseRegion(MlirBytecodeStream *pp,
                                      MlirIRSectionStack *stack, void *state,
                                      bool isIsolated) {
  uint64_t numBlocks;
  if (mlirBytecodeFailed(parseVarInt(pp, &numBlocks)))
    return mlirBytecodeEmitError("invalid number of blocks");
  if (numBlocks == 0)
    return mlirBytecodeSuccess();

  uint64_t numValues;
  if (mlirBytecodeFailed(parseVarInt(pp, &numValues)))
    return mlirBytecodeEmitError("invalid/missing number of values in region");

  mlirIRSectionStackDec(stack);
  if (mlirBytecodeFailed(
          mlirBytecodeRegionEnter(state, isIsolated, numBlocks, numValues)))
    return mlirBytecodeFailure();

  mlirIRSectionStackPushBack(isIsolated ? kStackBlockIsolated : kStackBlock,
                             numBlocks, stack);
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus parseOperation(MlirBytecodeStream *pp,
                                         MlirIRSectionStack *stack,
                                         void *callerState) {
  uint64_t name;
  if (mlirBytecodeFailed(parseVarInt(pp, &name)))
    return mlirBytecodeEmitError("invalid operation");

  uint8_t encodingMask;
  if (mlirBytecodeFailed(parseByte(pp, &encodingMask)))
    return mlirBytecodeEmitError("invalid encoding mask");

  MlirBytecodeAttrHandle loc;
  if (mlirBytecodeFailed(parseVarInt(pp, (uint64_t *)&loc)))
    return mlirBytecodeEmitError("invalid operation location");

  MlirBytecodeAttrHandle attrDict = {.id = kMlirBytecodeHandleSentinel};
  if ((encodingMask & kHasAttrs) &&
      mlirBytecodeFailed(parseVarInt(pp, (uint64_t *)&attrDict))) {
    return mlirBytecodeEmitError("invalid op attribute handle");
  }

  // Parsing all the variadic sizes.
  // This could have been done by using additional memory instead of
  // reparsing, but keeping number of allocations to minimum.
  uint64_t numResults = 0;
  const uint8_t *start = pp->pos;
  if ((encodingMask & kHasResults)) {
    if (mlirBytecodeSucceeded(parseVarInt(pp, &numResults))) {
      start = pp->pos;
      if (mlirBytecodeFailed(skipVarInts(pp, numResults)))
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
    if (mlirBytecodeSucceeded(parseVarInt(pp, &numOperands))) {
      start = pp->pos;
      if (mlirBytecodeFailed(skipVarInts(pp, numOperands)))
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
    if (mlirBytecodeSucceeded(parseVarInt(pp, &numSuccessors))) {
      start = pp->pos;
      if (mlirBytecodeFailed(skipVarInts(pp, numSuccessors)))
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
  if ((encodingMask & kHasInlineRegions)) {
    if (mlirBytecodeFailed(
            parseVarIntWithFlag(pp, &numRegions, &isIsolatedFromAbove))) {
      return mlirBytecodeEmitError("invalid number of regions");
    }
  }

  mlirIRSectionStackDec(stack);
  if (mlirBytecodeFailed(mlirBytecodeOperation(
          callerState, (MlirBytecodeOpHandle){.id = name}, attrDict,
          (MlirBytecodeStream *)&resultTypes, (MlirBytecodeStream *)&operands,
          (MlirBytecodeStream *)&successors, isIsolatedFromAbove, numRegions)))
    return mlirBytecodeEmitError("op function callback failed");

  if (numRegions > 0) {
    mlirIRSectionStackPushBack(isIsolatedFromAbove ? kStackRegionIsolated
                                                   : kStackRegion,
                               numRegions, stack);
  }

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeParseIRSection(MlirBytecodeFile *mlirFile,
                                              void *callerState) {
  const MlirBytecodeBytesRef irSection = getSection(mlirFile, kIR);
  MlirBytecodeStream pp = populateParserPosForSection(irSection);

  bool hasArgs;
  uint64_t numOps;
  if (mlirBytecodeFailed(parseVarIntWithFlag(&pp, &numOps, &hasArgs)))
    return mlirBytecodeFailure();
  if (hasArgs)
    return mlirBytecodeEmitError("IR section may not have block args");
  if (numOps != 1)
    return mlirBytecodeEmitError("only one top-level op supported");

  // TODO: Make this an arg.
  MlirIRSectionStack stack = {.maxDepth = 16, .top = -1};
  mlirIRSectionStackPushBack(kStackOperation, numOps, &stack);

  while (stack.top != -1) {
    uint64_t top = stack.data[stack.top];

    IRStackType cur = top & ((1 << irStackShift) - 1);

    if ((top >> irStackShift) == 0) {
      if (cur == kStackOperation &&
          mlirBytecodeFailed(mlirBytecodeBlockExit(callerState)))
        return mlirBytecodeFailure();
      if ((cur == kStackBlock || cur == kStackBlockIsolated) &&
          mlirBytecodeFailed(
              mlirBytecodeRegionExit(callerState, cur == kStackBlockIsolated)))
        return mlirBytecodeFailure();
      if ((cur == kStackRegion || cur == kStackRegionIsolated) &&
          mlirBytecodeFailed(mlirBytecodeIsolatedOperationExit(callerState)))
        return mlirBytecodeFailure();
      --stack.top;
      continue;
    }

    switch (cur) {
    case kStackBlock:
    case kStackBlockIsolated:
      if (mlirBytecodeFailed(parseBlock(&pp, &stack, callerState)))
        return mlirBytecodeFailure();
      break;
    case kStackOperation:
      if (mlirBytecodeFailed(parseOperation(&pp, &stack, callerState)))
        return mlirBytecodeFailure();
      break;
    case kStackRegionIsolated:
    case kStackRegion:
      if (mlirBytecodeFailed(parseRegion(&pp, &stack, callerState,
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
  MlirBytecodeStream pp = populateParserPosForSection(bytes);

  uint8_t magic[4];
  if (mlirBytecodeFailed(parseBytes(&pp, magic, 4)))
    return mlirBytecodeEmitError("unabled to read 4 byte magic code"), ret;
  if (magic[0] != 'M' || magic[1] != 'L' || magic[2] != 0xef || magic[3] != 'R')
    return mlirBytecodeEmitError("invalid file magic code"), ret;

  // Parse the bytecode version and producer.
  MlirBytecodeBytesRef producer;
  if (mlirBytecodeFailed(parseVarInt(&pp, &mlirFile->version)) ||
      mlirBytecodeFailed(parseNullTerminatedString(&pp, &producer)))
    return mlirBytecodeEmitError("invalid version or producer"), ret;
  mlirBytecodeEmitDebug("Producer: %s\n", producer.data);

  while (!empty(&pp)) {
    // Read the next section from the bytecode.
    if (mlirBytecodeFailed(parseSections(&pp, mlirFile->sectionData)))
      return mlirBytecodeEmitError("invalid section"), ret;
  }

  // Check that all of the required sections were found
  for (int i = 0; i < kNumSections; ++i) {
    if (getSection(&ret, i).data == NULL && !isSectionOptional(i)) {
      return mlirBytecodeEmitError("missing data for top-level section: ",
                                   sectionIDToString(i)),
             ret;
    }
  }
  // Mark as valid.
  mlirFile->producer = producer;
  return ret;
}

MlirBytecodeStatus parseMlirFile(MlirBytecodeBytesRef bytes, void *callerState,
                                 MlirBytecodeAttrCallBack attrFn,
                                 MlirBytecodeTypeCallBack typeFn,
                                 MlirBytecodeDialectCallBack dialectFn,
                                 MlirBytecodeDialectOpCallBack dialectOpFn,
                                 MlirBytecodeResourceCallBack resourceFn,
                                 MlirBytecodeStringCallBack stringFn) {
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
          mlirBytecodeForEachString(&mlirFile, callerState, stringFn)) ||
      // Process the dialect section.
      mlirBytecodeFailed(mlirBytecodeParseDialectSection(
          &mlirFile, callerState, dialectFn, dialectOpFn)) ||
      // Process the resource section if present.
      mlirBytecodeFailed(mlirBytecodeParseResourceSection(
          &mlirFile, callerState, resourceFn)) ||
      // Process the attribute and type section.
      mlirBytecodeFailed(mlirBytecodeForEachAttributeAndType(
          &mlirFile, callerState, attrFn, typeFn)) ||
      // Finally, process the IR section.
      mlirBytecodeFailed(mlirBytecodeParseIRSection(&mlirFile, callerState)))
    return mlirBytecodeFailure();

  return mlirBytecodeSuccess();
}

// ---

// Helper struct & function for iterating until specific value.
struct IterateStruct {
  MlirBytecodeBytesRef *ret;
  size_t n;
};

MlirBytecodeStatus iterateUntilN(void *state, MlirBytecodeStringHandle i,
                                 size_t total, MlirBytecodeBytesRef ref) {
  (void)total;
  struct IterateStruct *s = (struct IterateStruct *)state;
  if (i.id == s->n) {
    *s->ret = ref;
    return mlirBytecodeIterationInterupt();
  }
  return mlirBytecodeSuccess();
}

MlirBytecodeBytesRef
mlirBytecodeGetStringSectionValue(const MlirBytecodeFile *const mlirFile,
                                  MlirBytecodeStringHandle idx) {
  MlirBytecodeBytesRef ret = {.data = 0, .length = 0};
  struct IterateStruct callerState = {.ret = &ret, .n = idx.id};
  MlirBytecodeStatus status =
      mlirBytecodeForEachString(&callerState, mlirFile, iterateUntilN);
  assert(mlirBytecodeSucceeded(status));
  return ret;
}

MlirBytecodeOpRef
mlirBytecodeGetInstructionString(const MlirBytecodeFile *const mlirFile,
                                 MlirBytecodeOpHandle hdl) {
  const MlirBytecodeBytesRef dialectSection = getSection(mlirFile, kDialect);
  MlirBytecodeStream pp = populateParserPosForSection(dialectSection);

  MlirBytecodeOpRef ret = {.dialect = -1, .op = -1};
  uint64_t numDialects, t;
  if (mlirBytecodeFailed(parseVarInt(&pp, &numDialects)))
    return mlirBytecodeEmitError("unable to parse number of dialects"), ret;

  // TODO: Change to skipVarInts.
  for (uint64_t i = 0; i < numDialects; ++i)
    if (mlirBytecodeFailed(parseVarInt(&pp, &t)))
      return ret;

  uint64_t index = 0;
  while (!empty(&pp)) {
    uint64_t dialect, numOpNames;
    if (mlirBytecodeFailed(parseVarInt(&pp, &dialect)) ||
        mlirBytecodeFailed(parseVarInt(&pp, &numOpNames)))
      return ret;

    for (uint64_t j = 0; j < numOpNames; ++j) {
      uint64_t opName;
      if (mlirBytecodeFailed(parseVarInt(&pp, &opName)))
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
