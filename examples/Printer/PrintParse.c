//===-- PrintParse.c - Parser instantiation that just prints --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Example parser instantiation that just prints IR.
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Parsing inc configuration.
typedef struct MlirBytecodeOperationState MlirBytecodeOperationState;
typedef struct MlirBytecodeOperationState MlirBytecodeOperation;
// Include bytecode parsing implementation.
#include "mlirbcc/BytecodeTypes.h"
#include "mlirbcc/Parse.c.inc"
// Dialect attribute and types parsing hooks.
#include "mlirbcc/DialectBytecodeReader.c.inc"
// Dialects.
#include "mlirbcc/BuiltinParse.c.inc"

// Example that prints as one parses.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

// Define struct that captures operation state during parsing.
struct MlirBytecodeOperationState {
  MlirBytecodeOpHandle name;
  MlirBytecodeAttrHandle attrDict;
  MlirBytecodeLocHandle loc;
  bool isIsolated;
  bool hasOperands;
  bool hasRegions;
  bool hasResults;
  bool hasSuccessors;
};

typedef struct MlirMutableBytesRef MlirMutableBytesRef;
typedef MlirMutableBytesRef MlirBytecodeAttribute;
typedef MlirMutableBytesRef MlirBytecodeType;

// Just using some globals here for the example, this should be part of the
// user/instantiation state and globals avoided in general.

struct MlirMutableBytesRef {
  // Pointer to the start memory address.
  uint8_t *data;
  // Length of the fragment.
  size_t length;
};

// Effectively set up a cached dialect behavior.
#define MAX_DIALECTS 10
#define MAX_DEPTH 10

typedef struct {
  MlirBytecodeBytesRef bytes;
  MlirBytecodeDialectHandle dialectHandle;
  bool hasCustom;
} MlirBytecodeAttributeOrTypeRange;

struct ParsingState {
  MlirBytecodeStringHandle *dialectStr;

  MlirBytecodeAttrParseCallBack attrCallbacks[MAX_DIALECTS];
  MlirBytecodeTypeParseCallBack typeCallbacks[MAX_DIALECTS];

  // Mapping from string handle to section.
  MlirBytecodeBytesRef *strings;

  // Mapping from op handle to op.
  MlirBytecodeOpRef *ops;

  // Attribute handle to string representation.
  MlirBytecodeAttribute *attributes;
  // Mapping from attribute handle to range.
  MlirBytecodeAttributeOrTypeRange *attributeRange;

  // Type handle to string representation.
  MlirMutableBytesRef *types;
  // Mapping from type handle to range.
  MlirBytecodeAttributeOrTypeRange *typeRange;

  // Last used SSA at current depth.
  int ssaIdStack[MAX_DEPTH];

  // Number of values in regions nested.
  int regionNumValues[MAX_DEPTH];

  // Current depth.
  int depth;

  // Indent size for formatting.
  int indentSize;
};
typedef struct ParsingState ParsingState;

MlirBytecodeBytesRef getString(void *callerState,
                               MlirBytecodeStringHandle hdl) {
  ParsingState *state = callerState;
  assert(state->strings);
  return state->strings[hdl.id];
}

MlirBytecodeStatus
mlirBytecodeGetStringSectionValue(void *callerState,
                                  MlirBytecodeStringHandle idx,
                                  MlirBytecodeBytesRef *result) {
  *result = getString(callerState, idx);
  return mlirBytecodeSuccess();
}

MlirBytecodeOpRef mlirBytecodeGetOpName(void *callerState,
                                        MlirBytecodeOpHandle hdl) {
  ParsingState *state = callerState;
  assert(state->ops);
  return state->ops[hdl.id];
}

MlirBytecodeStatus mlirBytecodeParseAttribute(void *callerState,
                                              MlirBytecodeAttrHandle handle) {
  ParsingState *state = callerState;
  if (!state->attributes)
    return mlirBytecodeUnhandled();

  if (state->attributes[handle.id].length)
    return mlirBytecodeSuccess();

  MlirBytecodeAttributeOrTypeRange attr = state->attributeRange[handle.id];
  if (!attr.hasCustom) {
    int len = attr.bytes.length + sizeof("Textual()");
    state->attributes[handle.id].data = (uint8_t *)malloc(len);
    state->attributes[handle.id].length =
        snprintf((char *)state->attributes[handle.id].data, len,
                 "Textual(%.*s)", (int)attr.bytes.length, attr.bytes.data);

    mlirBytecodeEmitDebug("%.*s", (int)state->attributes[handle.id].length,
                          state->attributes[handle.id].data);
    return mlirBytecodeSuccess();
  }

  // Attribute parsing etc should happen here.
  if (attr.dialectHandle.id < MAX_DIALECTS &&
      state->attrCallbacks[attr.dialectHandle.id]) {
    MlirBytecodeStream stream = mlirBytecodeStreamCreate(attr.bytes);
    MlirBytecodeDialectReader reader = {.callerState = callerState,
                                        .stream = &stream};
    return state->attrCallbacks[attr.dialectHandle.id](&reader, handle);
  }

  mlirBytecodeEmitDebug("attr unhandled");
  return mlirBytecodeUnhandled();
}

MlirMutableBytesRef getAttribute(void *callerState,
                                 MlirBytecodeAttrHandle attr) {
  static char empty[] = "<<unknown>>";
  ParsingState *state = callerState;
  if (mlirBytecodeSucceeded(mlirBytecodeParseAttribute(callerState, attr)))
    return state->attributes[attr.id];
  return (MlirMutableBytesRef){.data = (uint8_t *)&empty[0],
                               .length = sizeof(empty)};
}

MlirBytecodeStatus mlirBytecodeParseType(void *callerState,
                                         MlirBytecodeTypeHandle handle) {
  ParsingState *state = callerState;
  if (!state->types)
    return mlirBytecodeUnhandled();

  if (state->types[handle.id].length)
    return mlirBytecodeSuccess();

  MlirBytecodeAttributeOrTypeRange type = state->typeRange[handle.id];
  mlirBytecodeEmitDebug("type %d %d", type.dialectHandle.id, handle.id);
  if (!type.hasCustom) {
    int len = type.bytes.length + sizeof("Textual()");
    state->types[handle.id].data = (uint8_t *)malloc(len);
    state->types[handle.id].length =
        snprintf((char *)state->types[handle.id].data, len, "Textual(%.*s)",
                 (int)type.bytes.length, type.bytes.data);

    mlirBytecodeEmitDebug("%.*s", (int)state->types[handle.id].length,
                          state->types[handle.id].data);
    return mlirBytecodeSuccess();
  }

  // Type parsing etc should happen here.
  if (type.dialectHandle.id < MAX_DIALECTS &&
      state->typeCallbacks[type.dialectHandle.id]) {
    MlirBytecodeStream stream = mlirBytecodeStreamCreate(type.bytes);
    MlirBytecodeDialectReader reader = {.callerState = callerState,
                                        .stream = &stream};
    return state->typeCallbacks[type.dialectHandle.id](&reader, handle);
  }

  return mlirBytecodeUnhandled();
}

MlirMutableBytesRef getType(void *callerState, MlirBytecodeTypeHandle type) {
  static char empty[] = "<<unknown>>";
  ParsingState *state = callerState;
  if (mlirBytecodeSucceeded(mlirBytecodeParseType(callerState, type)))
    return state->types[type.id];
  return (MlirMutableBytesRef){.data = (uint8_t *)&empty[0],
                               .length = sizeof(empty)};
}

// -----------------------------------
// Define dialect construction methods.

MlirBytecodeStatus mlirBytecodeCreateBuiltinFileLineColLoc(
    void *callerState, MlirBytecodeAttrHandle attrHandle,
    MlirBytecodeAttrHandle filename, uint64_t line, uint64_t col) {
  ParsingState *state = callerState;
  MlirBytecodeStatus ret = mlirBytecodeParseAttribute(callerState, filename);
  if (!mlirBytecodeSucceeded(ret))
    return ret;
  MlirMutableBytesRef str = getAttribute(callerState, filename);
  // Should avoid log here ...
  int len = str.length + sizeof("FileLineColLoc(::)") + ceil(log10(line + 1)) +
            ceil(log10(col + 1)) + 1;
  state->attributes[attrHandle.id].data = (uint8_t *)malloc(len);
  state->attributes[attrHandle.id].length =
      snprintf((char *)state->attributes[attrHandle.id].data, len,
               "FileLineColLoc(%.*s:%" PRId64 ":%" PRId64 ")", (int)str.length,
               str.data, line, col);

  mlirBytecodeEmitDebug("%s", state->attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinUnitAttr(void *callerState,
                                  MlirBytecodeAttrHandle attrHandle) {
  ParsingState *state = callerState;
  int len = sizeof("Unit");
  state->attributes[attrHandle.id].data = (uint8_t *)malloc(len);
  state->attributes[attrHandle.id].length =
      snprintf((char *)state->attributes[attrHandle.id].data, len, "Unit");

  mlirBytecodeEmitDebug("%s", state->attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinUnknownLoc(void *callerState,
                                    MlirBytecodeAttrHandle attrHandle) {
  ParsingState *state = callerState;
  int len = sizeof("UnknownLoc");
  state->attributes[attrHandle.id].data = (uint8_t *)malloc(len);
  state->attributes[attrHandle.id].length = snprintf(
      (char *)state->attributes[attrHandle.id].data, len, "UnknownLoc");

  mlirBytecodeEmitDebug("%s", state->attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinStringAttr(void *callerState,
                                    MlirBytecodeAttrHandle attrHandle,
                                    MlirBytecodeBytesRef str) {
  ParsingState *state = callerState;
  int len = str.length + 3;
  state->attributes[attrHandle.id].data = (uint8_t *)malloc(len);
  state->attributes[attrHandle.id].length =
      snprintf((char *)state->attributes[attrHandle.id].data, len, "\"%.*s\"",
               (int)str.length, str.data);

  mlirBytecodeEmitDebug("%s", state->attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeCreateBuiltinStringAttrWithType(
    void *callerState, MlirBytecodeAttrHandle bcAttrHandle,
    MlirBytecodeBytesRef str, MlirBytecodeTypeHandle type) {
  ParsingState *state = callerState;

  MlirMutableBytesRef typestr = getType(callerState, type);
  if (typestr.data == 0)
    return mlirBytecodeFailure();
  int len = str.length + 30;
  state->attributes[bcAttrHandle.id].data = (uint8_t *)malloc(len);
  state->attributes[bcAttrHandle.id].length = snprintf(
      (char *)state->attributes[bcAttrHandle.id].data, len, "\"%.*s, %.*s\"",
      (int)str.length, str.data, (int)typestr.length, typestr.data);

  mlirBytecodeEmitDebug("%s", state->attributes[bcAttrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinDictionaryAttr(void *callerState,
                                        MlirBytecodeAttrHandle attrHandle,
                                        MlirDictionaryHandleIterator *range) {
  ParsingState *state = callerState;
  int kMax = 200;
  state->attributes[attrHandle.id].data = (uint8_t *)malloc(kMax);
  int len = 0;

  len += snprintf((char *)state->attributes[attrHandle.id].data + len,
                  kMax - len, "{");
  MlirBytecodeAttrHandle name;
  MlirBytecodeAttrHandle value;
  while (mlirBytecodeSucceeded(mlirBytecodeGetNextDictionaryHandles(
      callerState, range, &name, &value))) {
    MlirBytecodeStatus ret = mlirBytecodeParseAttribute(callerState, name);
    if (!mlirBytecodeSucceeded(ret))
      return ret;
    ret = mlirBytecodeParseAttribute(callerState, value);
    if (!mlirBytecodeSucceeded(ret))
      return ret;
    MlirMutableBytesRef nameStr = state->attributes[name.id];
    MlirMutableBytesRef valueStr = state->attributes[value.id];
    len += snprintf((char *)state->attributes[attrHandle.id].data + len,
                    kMax - len, " %.*s = %.*s;", (int)nameStr.length,
                    nameStr.data, (int)valueStr.length, valueStr.data);
  }
  len += snprintf((char *)state->attributes[attrHandle.id].data + len,
                  kMax - len, " }");

  state->attributes[attrHandle.id].length = len;
  mlirBytecodeEmitDebug("%s", state->attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeCreateBuiltinIntegerAttr(
    void *callerState, MlirBytecodeAttrHandle attrHandle,
    MlirBytecodeTypeHandle typeHandle, uint64_t value) {
  ParsingState *state = callerState;
  char kMax = 30;
  state->attributes[attrHandle.id].data = (uint8_t *)malloc(kMax);
  state->attributes[attrHandle.id].length = snprintf(
      (char *)state->attributes[attrHandle.id].data, kMax,
      "Int(%.*s, %" PRId64 ")", (int)state->types[typeHandle.id].length,
      state->types[typeHandle.id].data, value);

  mlirBytecodeEmitDebug("%s", state->attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinTypeAttr(void *callerState,
                                  MlirBytecodeAttrHandle bcAttrHandle,
                                  MlirBytecodeTypeHandle value) {
  ParsingState *state = callerState;
  state->attributes[bcAttrHandle.id] = getType(callerState, value);

  mlirBytecodeEmitDebug("%s", state->attributes[bcAttrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinFloat32Type(void *callerState,
                                     MlirBytecodeTypeHandle typeHandle) {
  ParsingState *state = callerState;
  int len = sizeof("f32");
  state->types[typeHandle.id].data = (uint8_t *)malloc(len);
  state->types[typeHandle.id].length =
      snprintf((char *)state->types[typeHandle.id].data, len, "f32");

  mlirBytecodeEmitDebug("%s", state->types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinIndexType(void *callerState,
                                   MlirBytecodeTypeHandle typeHandle) {
  ParsingState *state = callerState;
  int len = sizeof("index");
  state->types[typeHandle.id].data = (uint8_t *)malloc(len);
  state->types[typeHandle.id].length =
      snprintf((char *)state->types[typeHandle.id].data, len, "index");

  mlirBytecodeEmitDebug("%s", state->types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeCreateBuiltinIntegerType(
    void *callerState, MlirBytecodeTypeHandle typeHandle,
    MlirBuiltinSignednessSemantics signedness, int width) {
  ParsingState *state = callerState;
  int len = ceil(log10(width)) + 4;
  state->types[typeHandle.id].data = (uint8_t *)malloc(len);
  int i = 0;
  if (signedness == kBuiltinIntegerTypeSigned) {
    state->types[typeHandle.id].data[i++] = 's';
  } else if (signedness == kBuiltinIntegerTypeUnsigned) {
    state->types[typeHandle.id].data[i++] = 'u';
  }
  state->types[typeHandle.id].length =
      snprintf((char *)state->types[typeHandle.id].data + i, len, "i%d",
               width) +
      i + 1;

  mlirBytecodeEmitDebug("%s", state->types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinFunctionType(void *callerState,
                                      MlirBytecodeTypeHandle typeHandle) {
  ParsingState *state = callerState;
  if (!state->types)
    return mlirBytecodeUnhandled();
  if (state->types[typeHandle.id].data)
    return mlirBytecodeSuccess();

  state->types[typeHandle.id].data = (uint8_t *)malloc(26);
  state->types[typeHandle.id].length = snprintf(
      (char *)state->types[typeHandle.id].data, 26, "function (...) -> (...)");

  mlirBytecodeEmitDebug("%s", state->types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinRankedTensorType(void *callerState,
                                          MlirBytecodeTypeHandle typeHandle) {
  ParsingState *state = callerState;
  if (!state->types)
    return mlirBytecodeUnhandled();
  if (state->types[typeHandle.id].data)
    return mlirBytecodeSuccess();

  state->types[typeHandle.id].data = (uint8_t *)malloc(20);
  state->types[typeHandle.id].length =
      snprintf((char *)state->types[typeHandle.id].data, 20, "tensor<...>");

  mlirBytecodeEmitDebug("%s", state->types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

// ----

MlirBytecodeStatus mlirBytecodeQueryBuiltinIntegerTypeWidth(
    void *callerState, MlirBytecodeTypeHandle typeHandle, unsigned *width) {
  ParsingState *state = callerState;
  if (!state->types)
    return mlirBytecodeUnhandled();

  MlirBytecodeStatus ret = mlirBytecodeParseType(callerState, typeHandle);
  if (!mlirBytecodeSucceeded(ret))
    return ret;
  MlirMutableBytesRef type = state->types[typeHandle.id];
  if (!type.length)
    return mlirBytecodeFailure();

  if (strncmp((char *)type.data, "index", type.length) == 0) {
    *width = 64;
    return mlirBytecodeSuccess();
  }

  char *c = (char *)type.data;
  for (unsigned i = 0; i < type.length; ++i) {
    if ('0' <= *c && *c <= '9')
      break;
    ++c;
  }
  int res = sscanf(c, "%u", width);
  return res == 1 ? mlirBytecodeSuccess() : mlirBytecodeFailure();
}

// ----

static MlirBytecodeStatus mlirBytecodeAttributesPush(void *callerState,
                                                     MlirBytecodeSize total) {
  ParsingState *state = callerState;
  // Note: this currently assumes that number of state->attributes don't change.
  if (state->attributes) {
    free(state->attributes);
  }

  state->attributes =
      (MlirMutableBytesRef *)malloc(total * sizeof(*state->attributes));
  memset(state->attributes, 0, total * sizeof(*state->attributes));
  state->attributeRange = (MlirBytecodeAttributeOrTypeRange *)malloc(
      total * sizeof(*state->attributeRange));
  mlirBytecodeEmitDebug("alloc'd state->attributes / %d", (int)total);
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus
mlirBytecodeAssociateAttributeRange(void *callerState,
                                    MlirBytecodeAttrHandle attrHandle,
                                    MlirBytecodeDialectHandle dialectHandle,
                                    MlirBytecodeBytesRef str, bool hasCustom) {
  ParsingState *state = callerState;
  state->attributeRange[attrHandle.id].bytes = str;
  state->attributeRange[attrHandle.id].hasCustom = hasCustom;
  state->attributeRange[attrHandle.id].dialectHandle = dialectHandle;
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus mlirBytecodeTypesPush(void *callerState,
                                                MlirBytecodeSize total) {
  ParsingState *state = callerState;
  if (state->types) {
    free(state->types);
  }
  state->types = (MlirMutableBytesRef *)malloc(total * sizeof(*state->types));
  memset(state->types, 0, total * sizeof(*state->types));
  state->typeRange = (MlirBytecodeAttributeOrTypeRange *)malloc(
      total * sizeof(*state->typeRange));
  mlirBytecodeEmitDebug("alloc'd state->types");
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus
mlirBytecodeAssociateTypeRange(void *callerState,
                               MlirBytecodeTypeHandle typeHandle,
                               MlirBytecodeDialectHandle dialectHandle,
                               MlirBytecodeBytesRef str, bool hasCustom) {
  ParsingState *state = callerState;
  state->typeRange[typeHandle.id].bytes = str;
  state->typeRange[typeHandle.id].hasCustom = hasCustom;
  state->typeRange[typeHandle.id].dialectHandle = dialectHandle;
  mlirBytecodeEmitDebug("type[%d] %d dialect = %d", typeHandle.id, hasCustom,
                        dialectHandle.id);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeDialectOpCallBack(void *callerState, MlirBytecodeOpHandle opHdl,
                              MlirBytecodeDialectHandle dialectHandle,
                              MlirBytecodeStringHandle stringHdl) {
  ParsingState *state = callerState;
  mlirBytecodeEmitDebug("\t\tdialect[%d] :: op[%d] = %d", (int)dialectHandle.id,
                        (int)opHdl.id, (int)stringHdl.id);
  const int total = 100;
  if (!state->ops) {
    // FIXME
    state->ops = malloc(total * sizeof(*state->ops));
    memset(state->ops, 0, total * sizeof(*state->ops));
  }
  if (opHdl.id >= total)
    return mlirBytecodeUnhandled();

  state->ops[opHdl.id] =
      (MlirBytecodeOpRef){.dialect = dialectHandle, .op = stringHdl};
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeDialectsPush(void *callerState,
                                            MlirBytecodeSize total) {
  ParsingState *state = callerState;
  if (state->dialectStr) {
    free(state->dialectStr);
  }
  state->dialectStr = malloc(total * sizeof(*state->dialectStr));
  memset(state->dialectStr, 0, total * sizeof(*state->dialectStr));
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeDialectCallBack(void *callerState,
                            MlirBytecodeDialectHandle dialectHandle,
                            MlirBytecodeStringHandle stringHdl) {
  ParsingState *state = callerState;
  MlirBytecodeBytesRef dialect = getString(callerState, stringHdl);
  mlirBytecodeEmitDebug("\t\tdialect[%d] = %s", (int)dialectHandle.id,
                        dialect.data);

  if (dialectHandle.id < MAX_DIALECTS) {
    if (strncmp((char *)dialect.data, "builtin", dialect.length) == 0) {
      mlirBytecodeEmitDebug("builtin handler set for %d", dialectHandle.id);
      state->attrCallbacks[dialectHandle.id] = &mlirBytecodeParseBuiltinAttr;
      state->typeCallbacks[dialectHandle.id] = &mlirBytecodeParseBuiltinType;
    }
  }

  state->dialectStr[dialectHandle.id] = stringHdl;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeStringsPush(void *callerState,
                                           MlirBytecodeSize numStrings) {
  ParsingState *state = callerState;
  if (state->strings) {
    free(state->strings);
  }
  state->strings =
      (MlirBytecodeBytesRef *)malloc(numStrings * sizeof(*state->strings));
  memset(state->strings, 0, numStrings * sizeof(*state->strings));
  mlirBytecodeEmitDebug("alloc'd state->strings");
  return mlirBytecodeSuccess();
}
MlirBytecodeStatus
mlirBytecodeAssociateStringRange(void *callerState,
                                 MlirBytecodeStringHandle hdl,
                                 MlirBytecodeBytesRef bytes) {
  ParsingState *state = callerState;
  state->strings[hdl.id] = bytes;
  return mlirBytecodeSuccess();
}

enum {
  /// Sentinel to indicate unset/unknown handle.
  kMlirBytecodeHandleSentinel = -1,
};
static bool hasAttrDict(MlirBytecodeAttrHandle attr) {
  return attr.id == (uint64_t)kMlirBytecodeHandleSentinel;
}

MlirBytecodeOperationState states[MLIRBC_IR_STACK_MAX_DEPTH + 1];
int stateDepth;

void mlirBytecodeIRSectionEnter(void *callerState, void *retBlock) {}

MlirBytecodeStatus mlirBytecodeOperationStatePush(
    void *callerState, MlirBytecodeOpHandle name, MlirBytecodeLocHandle loc,
    MlirBytecodeOperationStateHandle *opStateHandle) {
  MlirBytecodeOperationState *opState = &states[stateDepth++];
  mlirBytecodeEmitDebug("operation state push [depth = %d]", stateDepth);
  *opStateHandle = opState;
  opState->name = name;
  opState->loc = loc;
  opState->attrDict.id = kMlirBytecodeHandleSentinel;
  opState->hasOperands = false;
  opState->hasRegions = false;
  opState->hasResults = false;
  opState->hasSuccessors = false;
  opState->isIsolated = false;

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationBlockPush(void *callerState,
                               MlirBytecodeOperationHandle opHandle,
                               MlirBytecodeHandlesRef typeAndLocs) {
  ParsingState *state = callerState;
  bool first = true;
  printf("%*cblock", state->indentSize, '_');
  state->indentSize += 2;

  for (uint64_t i = 0; i < typeAndLocs.length; ++i) {
    MlirBytecodeTypeHandle type = typeAndLocs.handles[2 * i];
    // MlirBytecodeAttrHandle loc = typeAndLocs.handles[2 * i + 1];
    if (first) {
      printf("(");
    }
    // Note: block args locs push the print form too much.
    if (first)
      printf("%%%d : %s", state->ssaIdStack[state->depth]++,
             getType(callerState, type).data);
    else
      printf(", %%%d : %s", state->ssaIdStack[state->depth]++,
             getType(callerState, type).data);
    first = false;
  }

  if (!first)
    printf(")");
  printf(" ::\n");

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationBlockPop(void *callerState,
                              MlirBytecodeOperationHandle opHandle) {
  ParsingState *state = callerState;
  state->indentSize -= 2;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationRegionPush(void *callerState,
                                MlirBytecodeOperationHandle opHandle,
                                size_t numBlocks, size_t numValues) {
  ParsingState *state = callerState;
  MlirBytecodeOperationState *opState = opHandle;
  if (state->depth + 1 == MAX_DEPTH)
    return mlirBytecodeUnhandled();

  if (opState->isIsolated)
    state->ssaIdStack[++state->depth] = 0;
  else {
    state->ssaIdStack[state->depth + 1] =
        state->regionNumValues[state->depth] +
        (state->depth > 1 ? state->ssaIdStack[state->depth - 1] : 0);
    ++state->depth;
  }
  state->regionNumValues[state->depth] = numValues;
  state->indentSize += 2;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationRegionPop(void *callerState,
                               MlirBytecodeOperationStateHandle opStateHandle) {
  ParsingState *state = callerState;
  --state->depth;
  state->indentSize -= 2;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddAttributeDictionary(
    void *callerState, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeAttrHandle attrs) {
  MlirBytecodeOperationState *opState = opStateHandle;
  opState->attrDict = attrs;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus printOperationPrefix(void *callerState,
                                        MlirBytecodeOperationState *opState,
                                        MlirBytecodeHandlesRef retTypes) {
  ParsingState *state = callerState;
  MlirBytecodeOpRef opRef = mlirBytecodeGetOpName(callerState, opState->name);
  MlirBytecodeBytesRef opName = getString(callerState, opRef.op);
  MlirBytecodeBytesRef dialectName = getString(callerState, opRef.dialect);

  bool first = true;
  printf("%*c", state->indentSize, ' ');

  first = true;
  for (int i = 0, e = retTypes.length; i < e; ++i) {
    if (i != 0)
      printf(", ");

    printf("%%%d", state->ssaIdStack[state->depth]++);
    MlirBytecodeStatus ret =
        mlirBytecodeParseType(callerState, retTypes.handles[i]);
    if (mlirBytecodeSucceeded(ret)) {
      printf(":%.*s", (int)state->types[retTypes.handles[i].id].length,
             state->types[retTypes.handles[i].id].data);
    } else if (mlirBytecodeFailed(ret))
      return ret;
    else
      printf(":<<unknown type>>");
    first = false;
  }
  if (!first)
    printf(" = ");

  printf("%.*s.%.*s", (int)dialectName.length, dialectName.data,
         (int)opName.length, opName.data);

  if (!hasAttrDict(opState->attrDict)) {
    MlirBytecodeStatus ret =
        mlirBytecodeParseAttribute(callerState, opState->attrDict);
    if (mlirBytecodeFailed(ret)) {
      return ret;
    } else if (mlirBytecodeSucceeded(ret)) {
      MlirMutableBytesRef attrDictVal = state->attributes[opState->attrDict.id];
      printf(" %.*s ", (int)attrDictVal.length, attrDictVal.data);
    } else {
      printf(" << unhandled >> ");
    }
  }

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddResultTypes(
    void *callerState, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeHandlesRef ref) {
  MlirBytecodeOperationState *opState = opStateHandle;
  opState->hasResults = true;
  return printOperationPrefix(callerState, opState, ref);
}

MlirBytecodeStatus mlirBytecodeOperationStateAddOperands(
    void *callerState, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeHandlesRef ref) {
  MlirBytecodeOperationState *opState = opStateHandle;
  if (!opState->hasResults)
    if (!mlirBytecodeSucceeded(printOperationPrefix(
            callerState, opState,
            (MlirBytecodeHandlesRef){.handles = NULL, .length = 0})))
      return mlirBytecodeSuccess();
  opState->hasOperands = true;

  bool first = true;
  printf("(");
  for (uint64_t i = 0; i < ref.length; ++i) {
    if (first)
      printf("%%%d", (int)ref.handles[i].id);
    else
      printf(", %%%d", (int)ref.handles[i].id);
    first = false;
  }
  printf(")");

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddRegions(
    void *callerState, MlirBytecodeOperationStateHandle opStateHandle,
    uint64_t n, bool isIsolatedFromAbove) {
  MlirBytecodeOperationState *opState = opStateHandle;
  mlirBytecodeEmitDebug("numRegions = %d", (int)n);
  if (!opState->hasResults && !opState->hasOperands)
    if (!mlirBytecodeSucceeded(printOperationPrefix(
            callerState, opState,
            (MlirBytecodeHandlesRef){.handles = NULL, .length = 0})))
      return mlirBytecodeSuccess();
  opState->hasRegions = true;
  opState->isIsolated = isIsolatedFromAbove;
  printf("\n");
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddSuccessors(
    void *callerState, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeSizesRef ref) {
  MlirBytecodeOperationState *opState = opStateHandle;
  uint64_t numSuccessors = ref.length;
  if (!opState->hasResults && !opState->hasOperands && !opState->hasRegions)
    if (!mlirBytecodeSucceeded(printOperationPrefix(
            callerState, opState,
            (MlirBytecodeHandlesRef){.handles = NULL, .length = 0})))
      return mlirBytecodeSuccess();
  if (numSuccessors > 0) {
    printf(" // successors:");
    for (uint64_t i = 0; i < numSuccessors; ++i)
      printf(" ^%" PRIu64, ref.sizes[i]);
  }
  opState->hasSuccessors = true;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationStatePop(void *callerState,
                              MlirBytecodeOperationStateHandle opStateHandle,
                              MlirBytecodeOperationHandle *opHandle) {
  MlirBytecodeOperationState *opState = opStateHandle;
  mlirBytecodeEmitDebug("operation state pop [from depth = %d]", stateDepth);
  --stateDepth;
  if (!opState->hasResults && !opState->hasOperands && !opState->hasRegions &&
      !opState->hasSuccessors)
    if (!mlirBytecodeSucceeded(printOperationPrefix(
            callerState, opState,
            (MlirBytecodeHandlesRef){.handles = NULL, .length = 0})))
      return mlirBytecodeSuccess();
  if (!opState->hasRegions)
    printf("\n");
  *opHandle = opStateHandle;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeResourceSectionEnter(void *callerState,
                                 MlirBytecodeSize numExternalResourceGroups) {
  printf("-- Number of resource groups: %d\n", (int)numExternalResourceGroups);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeResourceGroupEnter(void *callerState,
                               MlirBytecodeStringHandle groupKey,
                               MlirBytecodeSize numResources) {
  printf("\tgroup '%s'\n", getString(callerState, groupKey).data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeResourceBlobCallBack(
    void *callerState, MlirBytecodeStringHandle groupKey,
    MlirBytecodeStringHandle resourceKey, MlirBytecodeBytesRef blob) {
  printf("\t\tblob size(resource %s. %s) = %" PRIu64 "\n",
         getString(callerState, groupKey).data,
         getString(callerState, resourceKey).data, blob.length);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeResourceBoolCallBack(
    void *callerState, MlirBytecodeStringHandle groupKey,
    MlirBytecodeStringHandle resourceKey, uint8_t boolResource) {
  printf("\t\tbool resource %s. %s = %d\n",
         getString(callerState, groupKey).data,
         getString(callerState, resourceKey).data, boolResource);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeResourceStringCallBack(
    void *callerState, MlirBytecodeStringHandle groupKey,
    MlirBytecodeStringHandle resourceKey, MlirBytecodeStringHandle strHandle) {
  printf("\t\tstring resource %s. %s = %s\n",
         getString(callerState, groupKey).data,
         getString(callerState, resourceKey).data,
         getString(callerState, strHandle).data);
  return mlirBytecodeSuccess();
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("usage: %s file.mlirbc\n", argv[0]);
    return 1;
  }

  int fd = open(argv[1], O_RDONLY);
  if (fd == -1) {
    return mlirBytecodeEmitError(NULL, "MlirBytecodeFailed to open file '%s'",
                                 argv[1]),
           1;
  }

  struct stat fileInfo;
  if (fstat(fd, &fileInfo) == -1) {
    return mlirBytecodeEmitError(NULL, "error getting the file size"), 1;
  }

  uint8_t *stream =
      (uint8_t *)mmap(0, fileInfo.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (stream == MAP_FAILED) {
    close(fd);
    return mlirBytecodeEmitError(NULL, "error mmapping the file"), 1;
  }

  // Set parsing state
  ParsingState state;
  state.attributes = 0;
  state.depth = 0;
  state.dialectStr = 0;
  state.indentSize = 0;
  state.ops = 0;
  state.regionNumValues[0] = 0;
  state.ssaIdStack[0] = 0;
  state.strings = 0;
  state.types = 0;

  MlirBytecodeBytesRef ref = {.data = stream, .length = fileInfo.st_size};
  MlirBytecodeParserState parserState =
      mlirBytecodePopulateParserState(&state, ref, NULL, 0);

  if (!mlirBytecodeParserStateEmpty(&parserState) &&
      mlirBytecodeFailed(mlirBytecodeParse(&state, &parserState, NULL)))
    return mlirBytecodeEmitError(NULL, "MlirBytecodeFailed to parse file"), 1;

  mlirBytecodeParseAttribute(&state, (MlirBytecodeAttrHandle){.id = 2});

  if (munmap(stream, fileInfo.st_size) == -1) {
    close(fd);
    perror("Error un-mmapping the file");
    exit(EXIT_FAILURE);
  }
  return 0;
}

#pragma GCC diagnostic pop
