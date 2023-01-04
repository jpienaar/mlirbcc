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

#include "mlirbcc/BytecodeTypes.h"

// Define struct that captures operation state during parsing.
struct MlirBytecodeOperationState {
  MlirBytecodeOpHandle name;
  MlirBytecodeAttrHandle attrDict;
  MlirBytecodeLocHandle loc;
  MlirBytecodeStream resultTypes;
  MlirBytecodeStream operands;
  bool isIsolated;
  bool hasRegions;
};
typedef struct MlirBytecodeOperationState MlirBytecodeOperationState;

#define MLIRBC_PARSE_IMPLEMENTATION
#include "mlirbcc/Parse.h"
// Dialects.
#include "mlirbcc/BuiltinParse.h"

// Example that prints as one parses. Note: these are not sufficient as one
// effectively needs to be able to lazily decode: ops can have attributes,
// attributes can have types, and the parsing of attribute or type can depend on
// a parsed attribute or type. But this is more of an example.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

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

MlirBytecodeFile *refFile;

// Effectively set up a cached dialect behavior.
#define MAX_DIALECTS 10
MlirBytecodeAttrParseCallBack attrCallbacks[MAX_DIALECTS];
MlirBytecodeTypeParseCallBack typeCallbacks[MAX_DIALECTS];

typedef struct {
  MlirBytecodeBytesRef bytes;
  MlirBytecodeDialectHandle dialectHandle;
  bool hasCustom;
} MlirBytecodeAttributeOrTypeRange;

MlirBytecodeAttribute *attributes = 0;
MlirBytecodeAttributeOrTypeRange *attributeRange = 0;
MlirMutableBytesRef *strings = 0;
MlirMutableBytesRef *types = 0;
MlirBytecodeAttributeOrTypeRange *typeRange = 0;

int ssaIdStack[10];
int regionNumValues[10];
int depth;

MlirMutableBytesRef getAttribute(MlirBytecodeAttrHandle attr) {
  static char empty[] = "<<unknown>>";
  if (attributes && attributes[attr.id].length)
    return attributes[attr.id];
  mlirBytecodeEmitDebug("unknown attribute %d", (int)attr.id);
  return (MlirMutableBytesRef){.data = (uint8_t *)&empty[0],
                               .length = sizeof(empty)};
}

MlirMutableBytesRef getType(MlirBytecodeTypeHandle type) {
  static char empty[] = "<<unknown>>";
  if (types && types[type.id].length)
    return types[type.id];
  return (MlirMutableBytesRef){.data = (uint8_t *)&empty[0],
                               .length = sizeof(empty)};
}

MlirBytecodeStatus mlirBytecodeProcessAttribute(void *callerState,
                                                MlirBytecodeAttrHandle handle) {
  if (attributes[handle.id].length)
    return mlirBytecodeSuccess();

  MlirBytecodeAttributeOrTypeRange attr = attributeRange[handle.id];
  if (!attr.hasCustom) {
    int len = attr.bytes.length + sizeof("Custom()");
    attributes[handle.id].data = (uint8_t *)malloc(len);
    attributes[handle.id].length =
        snprintf((char *)attributes[handle.id].data, len, "Custom(%.*s)",
                 (int)attr.bytes.length, attr.bytes.data);

    mlirBytecodeEmitDebug("%.*s", (int)attributes[handle.id].length,
                          attributes[handle.id].data);
    return mlirBytecodeSuccess();
  }

  // Attribute parsing etc should happen here.
  if (attr.dialectHandle.id < MAX_DIALECTS &&
      attrCallbacks[attr.dialectHandle.id]) {
    return attrCallbacks[attr.dialectHandle.id](callerState, handle, attr.bytes,
                                                attr.hasCustom);
  }

  mlirBytecodeEmitDebug("attr unhandled");
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeProcessType(void *callerState,
                                           MlirBytecodeTypeHandle handle) {
  if (types[handle.id].length)
    return mlirBytecodeSuccess();

  MlirBytecodeAttributeOrTypeRange type = typeRange[handle.id];
  if (!type.hasCustom) {
    int len = type.bytes.length + sizeof("Custom()");
    types[handle.id].data = (uint8_t *)malloc(len);
    types[handle.id].length =
        snprintf((char *)types[handle.id].data, len, "Custom(%.*s)",
                 (int)type.bytes.length, type.bytes.data);

    mlirBytecodeEmitDebug("%.*s", (int)types[handle.id].length,
                          types[handle.id].data);
    return mlirBytecodeSuccess();
  }

  // Type parsing etc should happen here.
  if (type.dialectHandle.id < MAX_DIALECTS &&
      typeCallbacks[type.dialectHandle.id]) {
    return typeCallbacks[type.dialectHandle.id](callerState, handle, type.bytes,
                                                type.hasCustom);
  }

  return mlirBytecodeUnhandled();
}

// Define dialect construction methods.

#ifdef __cplusplus
extern "C" {
#endif

MlirBytecodeStatus mlirBytecodeCreateBuiltinFileLineColLoc(
    void *callerState, MlirBytecodeAttrHandle attrHandle,
    MlirBytecodeAttrHandle filename, uint64_t line, uint64_t col) {
  MlirBytecodeStatus ret = mlirBytecodeProcessAttribute(callerState, filename);
  if (!mlirBytecodeSucceeded(ret))
    return ret;
  MlirMutableBytesRef str = attributes[filename.id];
  // Should avoid log here ...
  int len = str.length + sizeof("FileLineColLoc(::)") + ceil(log10(line + 1)) +
            ceil(log10(col + 1)) + 1;
  attributes[attrHandle.id].data = (uint8_t *)malloc(len);
  attributes[attrHandle.id].length =
      snprintf((char *)attributes[attrHandle.id].data, len,
               "FileLineColLoc(%.*s:%" PRId64 ":%" PRId64 ")", (int)str.length,
               str.data, line, col);

  mlirBytecodeEmitDebug("%s", attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinUnknownLoc(void *callerState,
                                    MlirBytecodeAttrHandle attrHandle) {
  int len = sizeof("UnknownLoc");
  attributes[attrHandle.id].data = (uint8_t *)malloc(len);
  attributes[attrHandle.id].length =
      snprintf((char *)attributes[attrHandle.id].data, len, "UnknownLoc");

  mlirBytecodeEmitDebug("%s", attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinStrAttr(void *callerState,
                                 MlirBytecodeAttrHandle attrHandle,
                                 MlirBytecodeStringHandle strHdl) {
  MlirBytecodeBytesRef str =
      mlirBytecodeGetStringSectionValue(callerState, refFile, strHdl);
  int len = str.length + 3;
  attributes[attrHandle.id].data = (uint8_t *)malloc(len);
  attributes[attrHandle.id].length =
      snprintf((char *)attributes[attrHandle.id].data, len, "\"%.*s\"",
               (int)str.length, str.data);

  mlirBytecodeEmitDebug("%s", attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeCreateBuiltinStringAttrWithType(
    void *bcUserState, MlirBytecodeAttrHandle bcAttrHandle,
    MlirBytecodeStringHandle value, MlirBytecodeTypeHandle type) {
  (void)bcUserState;

  MlirMutableBytesRef typeStr = getType(type);
  if (typeStr.data == 0)
    return mlirBytecodeFailure();
  MlirBytecodeBytesRef str =
      mlirBytecodeGetStringSectionValue(bcUserState, refFile, value);
  int len = str.length + 30;
  attributes[bcAttrHandle.id].data = (uint8_t *)malloc(len);
  attributes[bcAttrHandle.id].length =
      snprintf((char *)attributes[bcAttrHandle.id].data, len, "\"%.*s, %.*s\"",
               (int)str.length, str.data, (int)typeStr.length, typeStr.data);

  mlirBytecodeEmitDebug("%s", attributes[bcAttrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinDictionaryAttr(void *callerState,
                                        MlirBytecodeAttrHandle attrHandle,
                                        MlirDictionaryHandleIterator *range) {
  int kMax = 200;
  attributes[attrHandle.id].data = (uint8_t *)malloc(kMax);
  int len = 0;

  len +=
      snprintf((char *)attributes[attrHandle.id].data + len, kMax - len, "{");
  MlirBytecodeAttrHandle name;
  MlirBytecodeAttrHandle value;
  while (mlirBytecodeSucceeded(
      mlirBytecodeGetNextDictionaryHandles(range, &name, &value))) {
    MlirBytecodeStatus ret = mlirBytecodeProcessAttribute(callerState, name);
    if (!mlirBytecodeSucceeded(ret))
      return ret;
    ret = mlirBytecodeProcessAttribute(callerState, value);
    if (!mlirBytecodeSucceeded(ret))
      return ret;
    MlirMutableBytesRef nameStr = attributes[name.id];
    MlirMutableBytesRef valueStr = attributes[value.id];
    len += snprintf((char *)attributes[attrHandle.id].data + len, kMax - len,
                    " %.*s = %.*s;", (int)nameStr.length, nameStr.data,
                    (int)valueStr.length, valueStr.data);
  }
  len +=
      snprintf((char *)attributes[attrHandle.id].data + len, kMax - len, " }");

  attributes[attrHandle.id].length = len;
  mlirBytecodeEmitDebug("%s", attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeCreateBuiltinIntegerAttr(
    void *callerState, MlirBytecodeAttrHandle attrHandle,
    MlirBytecodeTypeHandle typeHandle, uint64_t value) {
  char kMax = 30;
  attributes[attrHandle.id].data = (uint8_t *)malloc(kMax);
  attributes[attrHandle.id].length = snprintf(
      (char *)attributes[attrHandle.id].data, kMax, "Int(%.*s, %" PRId64 ")",
      (int)types[typeHandle.id].length, types[typeHandle.id].data, value);

  mlirBytecodeEmitDebug("%s", attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinTypeAttr(void *bcUserState,
                                  MlirBytecodeAttrHandle bcAttrHandle,
                                  MlirBytecodeTypeHandle value) {
  (void)bcUserState;
  attributes[bcAttrHandle.id] = getType(value);

  mlirBytecodeEmitDebug("%s", attributes[bcAttrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinFloat32Type(void *callerState,
                                     MlirBytecodeTypeHandle typeHandle) {
  int len = sizeof("f32");
  types[typeHandle.id].data = (uint8_t *)malloc(len);
  types[typeHandle.id].length =
      snprintf((char *)types[typeHandle.id].data, len, "f32");

  mlirBytecodeEmitDebug("%s", types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinIndexType(void *callerState,
                                   MlirBytecodeTypeHandle typeHandle) {
  int len = sizeof("index");
  types[typeHandle.id].data = (uint8_t *)malloc(len);
  types[typeHandle.id].length =
      snprintf((char *)types[typeHandle.id].data, len, "index");

  mlirBytecodeEmitDebug("%s", types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeCreateBuiltinIntegerType(
    void *callerState, MlirBytecodeTypeHandle typeHandle,
    MlirBuiltinSignednessSemantics signedness, int width) {
  if (!types)
    return mlirBytecodeUnhandled();

  if (types[typeHandle.id].data)
    return mlirBytecodeSuccess();

  int len = ceil(log10(width)) + 4;
  types[typeHandle.id].data = (uint8_t *)malloc(len);
  int i = 0;
  if (signedness == kBuiltinIntegerTypeSigned) {
    types[typeHandle.id].data[i++] = 's';
  } else if (signedness == kBuiltinIntegerTypeUnsigned) {
    types[typeHandle.id].data[i++] = 'u';
  }
  types[typeHandle.id].length =
      snprintf((char *)types[typeHandle.id].data + i, len, "i%d", width) + i +
      1;

  mlirBytecodeEmitDebug("%s", types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinFunctionType(void *callerState,
                                      MlirBytecodeTypeHandle typeHandle) {
  if (!types)
    return mlirBytecodeUnhandled();
  if (types[typeHandle.id].data)
    return mlirBytecodeSuccess();

  types[typeHandle.id].data = (uint8_t *)malloc(26);
  types[typeHandle.id].length = snprintf((char *)types[typeHandle.id].data, 26,
                                         "function (...) -> (...)");

  mlirBytecodeEmitDebug("%s", types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinRankedTensorType(void *callerState,
                                          MlirBytecodeTypeHandle typeHandle) {
  if (!types)
    return mlirBytecodeUnhandled();
  if (types[typeHandle.id].data)
    return mlirBytecodeSuccess();

  types[typeHandle.id].data = (uint8_t *)malloc(20);
  types[typeHandle.id].length =
      snprintf((char *)types[typeHandle.id].data, 20, "tensor<...>");

  mlirBytecodeEmitDebug("%s", types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

// ----

MlirBytecodeStatus mlirBytecodeQueryBuiltinIntegerTypeWidth(
    void *callerState, MlirBytecodeTypeHandle typeHandle, unsigned *width) {
  if (!types)
    return mlirBytecodeUnhandled();

  MlirBytecodeStatus ret = mlirBytecodeProcessType(callerState, typeHandle);
  if (!mlirBytecodeSucceeded(ret))
    return ret;
  MlirMutableBytesRef type = types[typeHandle.id];
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

#ifdef __cplusplus
}
#endif

// ----

static MlirBytecodeStatus
printUnknownAttrs(void *callerState, MlirBytecodeAttrHandle attrHandle,
                  size_t total, MlirBytecodeDialectHandle dialectHandle,
                  MlirBytecodeBytesRef str, bool hasCustom) {
  // Note: this currently assumes that number of attributes don't change.
  if (!attributes) {
    attributes = (MlirMutableBytesRef *)malloc(total * sizeof(*attributes));
    memset(attributes, 0, total * sizeof(*attributes));
    attributeRange = (MlirBytecodeAttributeOrTypeRange *)malloc(
        total * sizeof(*attributeRange));
    mlirBytecodeEmitDebug("alloc'd attributes / %d", (int)total);
  }

  mlirBytecodeEmitDebug("attr[%d]", attrHandle.id);
  attributeRange[attrHandle.id].bytes = str;
  attributeRange[attrHandle.id].hasCustom = hasCustom;
  attributeRange[attrHandle.id].dialectHandle = dialectHandle;
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus
printUnknownTypes(void *callerState, MlirBytecodeTypeHandle typeHandle,
                  size_t total, MlirBytecodeDialectHandle dialectHandle,
                  MlirBytecodeBytesRef str, bool hasCustom) {

  // Note: this currently assumes that number of types don't change.
  if (!types) {
    types = (MlirMutableBytesRef *)malloc(total * sizeof(*types));
    memset(types, 0, total * sizeof(*types));
    typeRange =
        (MlirBytecodeAttributeOrTypeRange *)malloc(total * sizeof(*typeRange));
    mlirBytecodeEmitDebug("alloc'd types");
  }

  typeRange[typeHandle.id].bytes = str;
  typeRange[typeHandle.id].hasCustom = hasCustom;
  typeRange[typeHandle.id].dialectHandle = dialectHandle;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus printOpDialect(void *callerState, MlirBytecodeOpHandle opHdl,
                                  MlirBytecodeDialectHandle dialectHandle,
                                  MlirBytecodeStringHandle stringHdl) {
  mlirBytecodeEmitDebug("\t\tdialect[%d] :: op[%d] = %d", (int)dialectHandle.id,
                        (int)opHdl.id, (int)stringHdl.id);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus printDialect(void *callerState,
                                MlirBytecodeDialectHandle dialectHandle,
                                size_t total,
                                MlirBytecodeStringHandle stringHdl) {
  MlirBytecodeBytesRef dialect =
      mlirBytecodeGetStringSectionValue(callerState, refFile, stringHdl);
  mlirBytecodeEmitDebug("\t\tdialect[%d/%d] = %s", (int)dialectHandle.id,
                        (int)total, dialect.data);
  if (dialectHandle.id < MAX_DIALECTS) {
    if (strncmp((char *)dialect.data, "builtin", dialect.length) == 0) {
      attrCallbacks[dialectHandle.id] = &mlirBytecodeParseBuiltinAttr;
      typeCallbacks[dialectHandle.id] = &mlirBytecodeParseBuiltinType;
    }
  }

  // Mark as success even for now.
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus printStrings(void *callerState, MlirBytecodeStringHandle hdl,
                                size_t total, MlirBytecodeBytesRef bytes) {
  printf("\t\tstring[%d/%d] = %.*s\n", (int)hdl.id, (int)total,
         (int)bytes.length, bytes.data);

  return mlirBytecodeSuccess();
}

static int indentSize;

enum {
  /// Sentinel to indicate unset/unknown handle.
  kMlirBytecodeHandleSentinel = -1,
};
static bool hasAttrDict(MlirBytecodeAttrHandle attr) {
  return attr.id == (uint64_t)kMlirBytecodeHandleSentinel;
}

MlirBytecodeOperationState states[MlirIRSectionStackMaxDepth];
int stateDepth;

MlirBytecodeStatus mlirBytecodeOperationStatePush(
    void *callerState, MlirBytecodeOpHandle name, MlirBytecodeLocHandle loc,
    MlirBytecodeOperationStateHandle *opStateHandle) {
  MlirBytecodeOperationState *opState = &states[stateDepth++];
  opStateHandle->state = (void *)opState;
  opState->name = name;
  opState->loc = loc;
  opState->attrDict.id = kMlirBytecodeHandleSentinel;
  opState->hasRegions = false;
  opState->isIsolated = false;
  opState->resultTypes = (MlirBytecodeStream){.start = 0, .pos = 0, .end = 0};
  opState->operands = (MlirBytecodeStream){.start = 0, .pos = 0, .end = 0};

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateBlockPush(
    void *callerState, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeStream *stream, uint64_t numBlockArgs) {
  bool first = true;
  printf("%*cblock", indentSize, '_');
  indentSize += 2;

  for (uint64_t i = 0; i < numBlockArgs; ++i) {
    MlirBytecodeTypeHandle type;
    MlirBytecodeAttrHandle loc;
    if (!mlirBytecodeSucceeded(mlirBytecodeParseHandle(stream, &type)) ||
        !mlirBytecodeSucceeded(mlirBytecodeParseHandle(stream, &loc))) {
      return mlirBytecodeFailure();
    }
    if (first) {
      printf("(");
    }
    // Note: block args locs push the print form too much.
    if (first)
      printf("%%%d : %s", ssaIdStack[depth]++, getType(type).data);
    else
      printf(", %%%d : %s", ssaIdStack[depth]++, getType(type).data);
    first = false;
  }

  if (!first)
    printf(")");
  printf(" ::\n");

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeBlockPop(void *callerState,
                     MlirBytecodeOperationStateHandle opStateHandle) {
  indentSize -= 2;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeRegionPush(void *callerState,
                       MlirBytecodeOperationStateHandle opStateHandle,
                       size_t numBlocks, size_t numValues) {
  MlirBytecodeOperationState *opState = opStateHandle.state;
  if (opState->isIsolated)
    ssaIdStack[++depth] = 0;
  else {
    ssaIdStack[depth + 1] =
        (depth > 1 ? ssaIdStack[depth - 1] : 0) + regionNumValues[depth];
    ++depth;
  }
  regionNumValues[depth] = numValues;
  indentSize += 2;
  return mlirBytecodeSuccess();
}
MlirBytecodeStatus
mlirBytecodeRegionPop(void *callerState,
                      MlirBytecodeOperationStateHandle opStateHandle) {
  --depth;
  indentSize -= 2;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddAttributes(
    void *callerState, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeAttrHandle attrs) {
  MlirBytecodeOperationState *opState = opStateHandle.state;
  opState->attrDict = attrs;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddResultTypes(
    void *callerState, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeStream *stream, uint64_t numResults) {
  MlirBytecodeOperationState *opState = opStateHandle.state;
  opState->resultTypes.start = opState->resultTypes.pos = stream->pos;
  MlirBytecodeStatus ret = mlirBytecodeSkipHandles(stream, numResults);
  opState->resultTypes.end = stream->pos;
  return ret;
}

MlirBytecodeStatus mlirBytecodeOperationStateAddOperands(
    void *callerState, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeStream *stream, uint64_t numOperands) {
  MlirBytecodeOperationState *opState = opStateHandle.state;
  opState->operands.start = opState->operands.pos = stream->pos;
  MlirBytecodeStatus ret = mlirBytecodeSkipHandles(stream, numOperands);
  opState->operands.end = stream->pos;
  return ret;
}

MlirBytecodeStatus printOperationPrefix(void *callerState,
                                        MlirBytecodeOperationState *opState) {
  MlirBytecodeOpRef opRef =
      mlirBytecodeGetOpName(callerState, refFile, opState->name);
  MlirBytecodeBytesRef opName =
      mlirBytecodeGetStringSectionValue(callerState, refFile, opRef.op);
  MlirBytecodeBytesRef dialectName =
      mlirBytecodeGetStringSectionValue(callerState, refFile, opRef.dialect);

  bool first = true;
  printf("%*c", indentSize, ' ');

  MlirBytecodeTypeHandle retTy;
  first = true;
  while (mlirBytecodeSucceeded(
      mlirBytecodeParseHandle(&opState->resultTypes, &retTy))) {
    if (!first)
      printf(", ");

    printf("%%%d", ssaIdStack[depth]++);
    first = false;
  }
  if (!first)
    printf(" = ");

  printf("%.*s.%.*s", (int)dialectName.length, dialectName.data,
         (int)opName.length, opName.data);

  if (!hasAttrDict(opState->attrDict)) {
    MlirBytecodeStatus ret =
        mlirBytecodeProcessAttribute(callerState, opState->attrDict);
    if (!mlirBytecodeSucceeded(ret))
      return ret;
    MlirMutableBytesRef attrDictVal = attributes[opState->attrDict.id];
    printf(" %.*s ", (int)attrDictVal.length, attrDictVal.data);
  }

  MlirBytecodeOpHandle op;
  first = true;
  while (
      mlirBytecodeSucceeded(mlirBytecodeParseHandle(&opState->operands, &op))) {
    if (first)
      printf("(%%%" PRIu64, (uint64_t)op.id);
    else
      printf(", %%%" PRIu64, (uint64_t)op.id);
    first = false;
  }
  if (!first)
    printf(")");

  first = true;
  mlirBytecodeStreamReset(&opState->resultTypes);
  while (mlirBytecodeSucceeded(
      mlirBytecodeParseHandle(&opState->resultTypes, &retTy))) {
    if (first)
      printf(" : ");
    else
      printf(", ");

    MlirBytecodeStatus ret = mlirBytecodeProcessType(callerState, retTy);
    if (!mlirBytecodeSucceeded(ret))
      return ret;
    printf("%.*s", (int)types[retTy.id].length, types[retTy.id].data);

    first = false;
  }

  printf("\n");
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddRegions(
    void *callerState, MlirBytecodeOperationStateHandle opStateHandle,
    uint64_t n) {
  MlirBytecodeOperationState *opState = opStateHandle.state;
  opState->hasRegions = true;
  mlirBytecodeEmitDebug("numRegions = %d", (int)n);
  return printOperationPrefix(callerState, opState);
}

MlirBytecodeStatus mlirBytecodeOperationStateAddSuccessors(
    void *callerState, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeStream *stream, uint64_t numSuccessors) {
  if (numSuccessors > 0) {
    printf("// successors");
    for (uint64_t i = 0; i < numSuccessors; ++i) {
      uint64_t index;
      MlirBytecodeStatus ret = mlirBytecodeParseVarInt(stream, &index);
      if (!mlirBytecodeSucceeded(ret))
        return ret;
      printf(" ^%" PRIu64, index);
    }
    printf("\n");
  }
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationStatePop(void *callerState,
                              MlirBytecodeOperationStateHandle opStateHandle) {
  MlirBytecodeOperationState *opState = opStateHandle.state;
  mlirBytecodeEmitDebug("operation state pop");
  if (!opState->hasRegions)
    return printOperationPrefix(callerState, opState);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeIsolatedOperationExit(void *callerState,
                                                     bool isIsolated) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus printResourceDialect(
    void *callerState, MlirBytecodeStringHandle groupKey,
    MlirBytecodeSize totalGroups, MlirBytecodeStringHandle resourceKey,
    MlirBytecodeAsmResourceEntryKind kind, const MlirBytecodeBytesRef *blob,
    const uint8_t *boolResource, const MlirBytecodeStringHandle *strHandle) {
  switch (kind) {
  case kMlirBytecodeResourceEntryBlob:
    printf(
        "\t\tblob size(resource %s.%s) = %ld\n",
        mlirBytecodeGetStringSectionValue(callerState, refFile, groupKey).data,
        mlirBytecodeGetStringSectionValue(callerState, refFile, resourceKey)
            .data,
        blob->length);
    break;
  case kMlirBytecodeResourceEntryBool:
    printf(
        "\t\tbool resource %s.%s = %d\n",
        mlirBytecodeGetStringSectionValue(callerState, refFile, groupKey).data,
        mlirBytecodeGetStringSectionValue(callerState, refFile, resourceKey)
            .data,
        *boolResource);
    break;
  case kMlirBytecodeResourceEntryString:
    printf(
        "\t\tstring resource %s.%s = %s\n",
        mlirBytecodeGetStringSectionValue(callerState, refFile, groupKey).data,
        mlirBytecodeGetStringSectionValue(callerState, refFile, resourceKey)
            .data,
        mlirBytecodeGetStringSectionValue(callerState, refFile, *strHandle)
            .data);
    break;
  }

  return mlirBytecodeSuccess();
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s file.mlirbc\n", argv[0]);
    return 1;
  }

  int fd = open(argv[1], O_RDONLY);
  if (fd == -1) {
    return mlirBytecodeEmitError("MlirBytecodeFailed to open file '%s'",
                                 argv[1]),
           1;
  }

  struct stat fileInfo;
  if (fstat(fd, &fileInfo) == -1) {
    return mlirBytecodeEmitError("error getting the file size"), 1;
  }

  uint8_t *stream =
      (uint8_t *)mmap(0, fileInfo.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (stream == MAP_FAILED) {
    close(fd);
    return mlirBytecodeEmitError("error mmapping the file"), 1;
  }

  MlirBytecodeBytesRef ref = {.data = stream, .length = fileInfo.st_size};
  MlirBytecodeFile mlirFile = mlirBytecodePopulateFile(ref);

  // Set global state
  refFile = &mlirFile;
  attributes = 0;
  types = 0;
  strings = 0;
  ssaIdStack[0] = 0;
  regionNumValues[0] = 0;
  depth = 0;
  bool populated = !mlirBytecodeFileEmpty(&mlirFile);
  if (!populated)
    goto end;

  fprintf(stderr, "Parsing strings\n");
  if (mlirBytecodeFailed(
          mlirBytecodeForEachString(NULL, refFile, &printStrings))) {
    return mlirBytecodeEmitError("MlirBytecodeFailed to parse dialect"), 1;
  }

  fprintf(stderr, "Parsing dialects\n");
  if (mlirBytecodeFailed(mlirBytecodeParseDialectSection(
          NULL, refFile, &printDialect, &printOpDialect))) {
    return mlirBytecodeEmitError("MlirBytecodeFailed to parse dialects"), 1;
  }

  fprintf(stderr, "Parsing attributes & types\n");
  if (mlirBytecodeFailed(mlirBytecodeForEachAttributeAndType(
          NULL, refFile, &printUnknownAttrs, &printUnknownTypes))) {
    return mlirBytecodeEmitError("MlirBytecodeFailed to parse attr/type"), 1;
  }

  fprintf(stderr, "Parsing resources\n");
  if (mlirBytecodeFailed(
          mlirBytecodeForEachResource(NULL, refFile, &printResourceDialect))) {
    return mlirBytecodeEmitError("MlirBytecodeFailed to parse resources"), 1;
  }

  fprintf(stderr, "Parsing IR\n");
  indentSize = 0;
  if (mlirBytecodeFailed(mlirBytecodeParseIRSection(NULL, refFile))) {
    return mlirBytecodeEmitError("MlirBytecodeFailed to parse IR"), 1;
  }

  printf("\n-------\n");

  // All wrapped up together.
#if 0
  if (mlirBytecodeFailed(mlirBytecodeParseFile(
          NULL, ref,&printUnknownAttrs, &printUnknownTypeDialect, &printDialect,
          &printOpDialect,
          &printResourceDialect, &printStrings)))
    return mlirBytecodeEmitError("MlirBytecodeFailed to parse file"), 1;

  if (mlirBytecodeFailed(mlirBytecodeParseFile(
           NULL, ref,&printUnknownAttrs, &printUnknownTypeDialect, &printDialect,
          &printOpDialect,
          &printResourceDialect, &printStrings)))
    return mlirBytecodeEmitError("MlirBytecodeFailed to parse file"), 1;
#endif

end:
  if (munmap(stream, fileInfo.st_size) == -1) {
    close(fd);
    perror("Error un-mmapping the file");
    exit(EXIT_FAILURE);
  }
  return populated;
}

#pragma GCC diagnostic pop
