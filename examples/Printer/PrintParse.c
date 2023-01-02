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
// a parsed attribute or type (one could iterator until fixed point ... but
// lazily decoding across these as needed seems much better).

// Extern functions defined here. This allows for the same parsing structure but
// customization per linked binary. Now, these also allow for passing an
// arbitrary state along, this state is to allow for caches and avoids globals.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

// Just using some globals here for the example, this should be part of the user
// state.

struct MlirMutableBytesRef {
  // Pointer to the start memory address.
  uint8_t *data;
  // Length of the fragment.
  size_t length;
};
typedef struct MlirMutableBytesRef MlirMutableBytesRef;

MlirBytecodeFile *refFile;

// Effectively set up a cached dialect behavior.
#define MAX_DIALECTS 10
MlirBytecodeAttrCallBack attrCallbacks[MAX_DIALECTS];
MlirBytecodeTypeCallBack typeCallbacks[MAX_DIALECTS];

MlirMutableBytesRef *attributes = 0;
MlirMutableBytesRef *strings = 0;
MlirMutableBytesRef *types = 0;

int ssaIdStack[10];
int depth;

static MlirMutableBytesRef getAttribute(MlirBytecodeAttrHandle attr) {
  static char empty[] = "<<unknown >>";
  if (!mlirBytecodeIsSentinel(attr) && attributes && attributes[attr.id].length)
    return attributes[attr.id];
  mlirBytecodeEmitDebug("unknown attribute %d", (int)attr.id);
  return (MlirMutableBytesRef){.data = (uint8_t *)&empty[0],
                               .length = sizeof(empty)};
}

static MlirMutableBytesRef getType(MlirBytecodeTypeHandle type) {
  static char empty[] = "<<unknown>>";
  if (types && types[type.id].length)
    return types[type.id];
  return (MlirMutableBytesRef){.data = (uint8_t *)&empty[0],
                               .length = sizeof(empty)};
}

// Define dialect construction methods.

#ifdef __cplusplus
extern "C" {
#endif

MlirBytecodeStatus mlirBytecodeCreateBuiltinFileLineColLoc(
    void *state, MlirBytecodeAttrHandle attrHandle,
    MlirBytecodeAttrHandle filename, uint64_t line, uint64_t col) {
  (void)state;

  MlirMutableBytesRef str =
      (attributes != 0 && attributes[filename.id].length != 0)
          ? attributes[filename.id]
          : (MlirMutableBytesRef){.length = 0};
  // Should avoid log here ...
  int len = str.length + sizeof("FileLineColLoc(::)") + ceil(log10(line + 1)) +
            ceil(log10(col + 1)) + 1;
  attributes[attrHandle.id].data = (uint8_t *)malloc(len);
  attributes[attrHandle.id].length =
      snprintf((char *)attributes[attrHandle.id].data, len,
               "FileLineColLoc(%.*s:%" PRId64 ":%" PRId64 ")", (int)str.length,
               str.data, line, col);

  mlirBytecodeEmitDebug("%s\n", attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinUnknownLoc(void *state,
                                    MlirBytecodeAttrHandle attrHandle) {
  (void)state;

  int len = sizeof("UnknownLoc");
  attributes[attrHandle.id].data = (uint8_t *)malloc(len);
  attributes[attrHandle.id].length =
      snprintf((char *)attributes[attrHandle.id].data, len, "UnknownLoc");

  mlirBytecodeEmitDebug("%s\n", attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinStrAttr(void *state, MlirBytecodeAttrHandle attrHandle,
                                 MlirBytecodeStringHandle strHdl) {
  (void)state;

  MlirBytecodeBytesRef str =
      mlirBytecodeGetStringSectionValue(state, refFile, strHdl);
  int len = str.length + 3;
  attributes[attrHandle.id].data = (uint8_t *)malloc(len);
  attributes[attrHandle.id].length =
      snprintf((char *)attributes[attrHandle.id].data, len, "\"%.*s\"",
               (int)str.length, str.data);

  mlirBytecodeEmitDebug("%s\n", attributes[attrHandle.id].data);
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

  mlirBytecodeEmitDebug("%s\n", attributes[bcAttrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinDictionaryAttr(void *state,
                                        MlirBytecodeAttrHandle attrHandle,
                                        MlirDictionaryHandleIterator *range) {
  (void)state;

  if (!attributes)
    return mlirBytecodeUnhandled();
  int kMax = 200;
  attributes[attrHandle.id].data = (uint8_t *)malloc(kMax);
  int len = 0;

  len +=
      snprintf((char *)attributes[attrHandle.id].data + len, kMax - len, "{");
  MlirBytecodeAttrHandle name;
  MlirBytecodeAttrHandle value;
  while (mlirBytecodeSucceeded(
      mlirBytecodeGetNextDictionaryHandles(range, &name, &value))) {
    MlirMutableBytesRef nameStr = getAttribute(name);
    if (nameStr.length == 0 || attributes[value.id].length == 0) {
      return mlirBytecodeUnhandled();
    }

    MlirMutableBytesRef valueStr =
        attributes ? attributes[value.id] : (MlirMutableBytesRef){.length = 0};
    len += snprintf((char *)attributes[attrHandle.id].data + len, kMax - len,
                    " %.*s = %.*s;", (int)nameStr.length, nameStr.data,
                    (int)valueStr.length, valueStr.data);
  }
  len +=
      snprintf((char *)attributes[attrHandle.id].data + len, kMax - len, " }");

  attributes[attrHandle.id].length = len;
  mlirBytecodeEmitDebug("%s\n", attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeCreateBuiltinIntegerAttr(
    void *state, MlirBytecodeAttrHandle attrHandle,
    MlirBytecodeTypeHandle typeHandle, uint64_t value) {
  (void)state;

  char kMax = 30;
  attributes[attrHandle.id].data = (uint8_t *)malloc(kMax);
  attributes[attrHandle.id].length = snprintf(
      (char *)attributes[attrHandle.id].data, kMax, "Int(%.*s, %" PRId64 ")",
      (int)types[typeHandle.id].length, types[typeHandle.id].data, value);

  mlirBytecodeEmitDebug("%s\n", attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinTypeAttr(void *bcUserState,
                                  MlirBytecodeAttrHandle bcAttrHandle,
                                  MlirBytecodeTypeHandle value) {
  (void)bcUserState;
  attributes[bcAttrHandle.id] = getType(value);

  mlirBytecodeEmitDebug("%s\n", attributes[bcAttrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinFloat32Type(void *state,
                                     MlirBytecodeTypeHandle typeHandle) {
  (void)state;

  int len = sizeof("f32");
  types[typeHandle.id].data = (uint8_t *)malloc(len);
  types[typeHandle.id].length =
      snprintf((char *)types[typeHandle.id].data, len, "f32");

  mlirBytecodeEmitDebug("%s\n", types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinIndexType(void *state,
                                   MlirBytecodeTypeHandle typeHandle) {
  (void)state;

  int len = sizeof("index");
  types[typeHandle.id].data = (uint8_t *)malloc(len);
  types[typeHandle.id].length =
      snprintf((char *)types[typeHandle.id].data, len, "index");

  mlirBytecodeEmitDebug("%s\n", types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeCreateBuiltinIntegerType(
    void *state, MlirBytecodeTypeHandle typeHandle,
    MlirBuiltinSignednessSemantics signedness, int width) {
  if (types && types[typeHandle.id].data)
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

  mlirBytecodeEmitDebug("%s\n", types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinFunctionType(void *state,
                                      MlirBytecodeTypeHandle typeHandle) {
  (void)state;
  if (types && types[typeHandle.id].data)
    return mlirBytecodeSuccess();

  types[typeHandle.id].data = (uint8_t *)malloc(26);
  types[typeHandle.id].length = snprintf((char *)types[typeHandle.id].data, 26,
                                         "function (...) -> (...)");

  mlirBytecodeEmitDebug("%s\n", types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinRankedTensorType(void *state,
                                          MlirBytecodeTypeHandle typeHandle) {
  if (types && types[typeHandle.id].data)
    return mlirBytecodeSuccess();

  types[typeHandle.id].data = (uint8_t *)malloc(20);
  types[typeHandle.id].length =
      snprintf((char *)types[typeHandle.id].data, 20, "tensor<...>");

  mlirBytecodeEmitDebug("%s\n", types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

// ----

MlirBytecodeStatus mlirBytecodeQueryBuiltinIntegerTypeWidth(
    void *state, MlirBytecodeTypeHandle typeHandle, unsigned *width) {
  (void)state;

  if (!types || types->length == 0)
    return mlirBytecodeUnhandled();

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
printUnknownAttrs(void *state, MlirBytecodeDialectHandle dialectHandle,
                  MlirBytecodeAttrHandle attrHandle, size_t total,
                  bool hasCustom, MlirBytecodeBytesRef str) {
  (void)state;
  // Note: this currently assumes that number of attributes don't change.
  if (!attributes) {
    attributes = (MlirMutableBytesRef *)malloc(total * sizeof(*attributes));
    memset(attributes, 0, total * sizeof(*attributes));
  }

  if (!mlirBytecodeIsSentinel(attrHandle) && attributes[attrHandle.id].length) {
    return mlirBytecodeSuccess();
  }

  mlirBytecodeEmitDebug("\t\tdialect[%" PRIu64 "] :: attr[%d/%d] ",
                        dialectHandle.id, (int)attrHandle.id, (int)total);

  // Attribute parsing etc should happen here.
  if (!hasCustom) {
    int len = str.length + sizeof("Custom()");
    attributes[attrHandle.id].data = (uint8_t *)malloc(len);
    attributes[attrHandle.id].length =
        snprintf((char *)attributes[attrHandle.id].data, len, "Custom(%.*s)",
                 (int)str.length, str.data);

    mlirBytecodeEmitDebug("%.*s\n", (int)attributes[attrHandle.id].length,
                          attributes[attrHandle.id].data);
    return mlirBytecodeSuccess();
  }

  if (dialectHandle.id < MAX_DIALECTS && attrCallbacks[dialectHandle.id])
    return attrCallbacks[dialectHandle.id](state, dialectHandle, attrHandle,
                                           total, hasCustom, str);

  return mlirBytecodeUnhandled();
}

static MlirBytecodeStatus
printUnknownTypeDialect(void *state, MlirBytecodeDialectHandle dialectHandle,
                        MlirBytecodeTypeHandle typeHandle, size_t total,
                        bool hasCustom, MlirBytecodeBytesRef str) {
  (void)state;
  if (!types) {
    types = (MlirMutableBytesRef *)malloc(total * sizeof(*types));
    memset(types, 0, total * sizeof(*types));
  }

  if (types[typeHandle.id].length)
    return mlirBytecodeSuccess();

  mlirBytecodeEmitDebug("\t\tdialect[%d] :: type[%d/%d] ",
                        (int)dialectHandle.id, (int)typeHandle.id, (int)total);

  if (!hasCustom) {
    int len = str.length + sizeof("Custom()");
    types[typeHandle.id].data = (uint8_t *)malloc(len);
    types[typeHandle.id].length =
        snprintf((char *)types[typeHandle.id].data, len, "Custom(%.*s)",
                 (int)str.length, str.data);

    mlirBytecodeEmitDebug("%.*s\n", (int)types[typeHandle.id].length,
                          types[typeHandle.id].data);
    return mlirBytecodeSuccess();
  }

  if (dialectHandle.id < MAX_DIALECTS && typeCallbacks[dialectHandle.id]) {
    MlirBytecodeStatus s = typeCallbacks[dialectHandle.id](
        state, dialectHandle, typeHandle, total, hasCustom, str);
    if (mlirBytecodeHandled(s))
      return s;
  }

  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus printOpDialect(void *state,
                                  MlirBytecodeDialectHandle dialectHandle,
                                  MlirBytecodeOpHandle opHdl,
                                  MlirBytecodeStringHandle stringHdl) {
  (void)state;

  mlirBytecodeEmitDebug("\t\tdialect[%d] :: op[%d] = %d\n",
                        (int)dialectHandle.id, (int)opHdl.id,
                        (int)stringHdl.id);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus printDialect(void *state,
                                MlirBytecodeDialectHandle dialectHandle,
                                size_t total,
                                MlirBytecodeStringHandle stringHdl) {
  MlirBytecodeBytesRef dialect =
      mlirBytecodeGetStringSectionValue(state, refFile, stringHdl);
  mlirBytecodeEmitDebug("\t\tdialect[%d/%d] = %s\n", (int)dialectHandle.id,
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

MlirBytecodeStatus printStrings(void *state, MlirBytecodeStringHandle hdl,
                                size_t total, MlirBytecodeBytesRef bytes) {
  (void)state;

  if (!types) {
    types = (MlirMutableBytesRef *)malloc(total * sizeof(*types));
    memset(types, 0, total * sizeof(*types));
  }
  printf("\t\tstring[%d/%d] = %.*s\n", (int)hdl.id, (int)total,
         (int)bytes.length, bytes.data);

  return mlirBytecodeSuccess();
}

static int indentSize;

MlirBytecodeStatus
mlirBytecodeOperationStatePush(void *callerState, MlirBytecodeOpHandle name,
                               MlirBytecodeLocHandle loc,
                               MlirBytecodeOperationState *opState) {
  opState->name = name;
  opState->loc = loc;
  opState->attrDict.id = kMlirBytecodeHandleSentinel;
  opState->hasRegions = false;
  opState->isIsolated = false;
  opState->resultTypes = (MlirBytecodeStream){.start = 0, .pos = 0, .end = 0};
  opState->operands= (MlirBytecodeStream){.start = 0, .pos = 0, .end = 0};

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeBlockEnter(void *state,
                                          MlirBlockArgHandleIterator *blockArgs,
                                          size_t numOps) {
  MlirBytecodeTypeHandle type;
  MlirBytecodeAttrHandle loc;
  bool first = true;
  printf("%*cblock", indentSize, '_');
  indentSize += 2;

  while (mlirBytecodeSucceeded(
      mlirBytecodeGetNextBlockArgHandles(blockArgs, &type, &loc))) {
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
MlirBytecodeStatus mlirBytecodeBlockExit(void *state,
                                         MlirBytecodeOperationState *opState) {
  indentSize -= 2;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeRegionEnter(void *state, size_t numBlocks, size_t numValues,
                        MlirBytecodeOperationState *opState) {
  if (opState->isIsolated)
    ssaIdStack[++depth] = 0;
  indentSize += 2;
  return mlirBytecodeSuccess();
}
MlirBytecodeStatus mlirBytecodeRegionExit(void *state,
                                          MlirBytecodeOperationState *opState) {
  if (opState->isIsolated)
    --depth;
  indentSize -= 2;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationStateAddAttributes(void *callerState,
                                        MlirBytecodeAttrHandle attrs,
                                        MlirBytecodeOperationState *opState) {
  opState->attrDict = attrs;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddResultTypes(
    void *callerState, MlirBytecodeStream *stream, uint64_t numResults,
    MlirBytecodeOperationState *opState) {
  opState->resultTypes.start = opState->resultTypes.pos = stream->pos;
  MlirBytecodeStatus ret = mlirBytecodeSkipHandles(stream, numResults);
  opState->resultTypes.end = stream->pos;
  return ret;
}

MlirBytecodeStatus mlirBytecodeOperationStateAddOperands(
    void *callerState, MlirBytecodeStream *stream, uint64_t numOperands,
    MlirBytecodeOperationState *opState) {
  opState->operands.start = opState->operands.pos = stream->pos;
  MlirBytecodeStatus ret = mlirBytecodeSkipHandles(stream, numOperands);
  opState->operands.end = stream->pos;
  return ret;
}

MlirBytecodeStatus
mlirBytecodeOperationStateAddBlocks(void *callerState, uint64_t numBlocks,
                                    MlirBytecodeOperationState *opState) {
  // NOP in printing as in order.
  return mlirBytecodeSuccess();
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
      mlirBytecodeReadHandle(&opState->resultTypes, &retTy))) {
    if (!first)
      printf(", ");

    printf("%%%d", ssaIdStack[depth]++);
    first = false;
  }
  if (!first)
    printf(" = ");

  printf("%.*s.%.*s", (int)dialectName.length, dialectName.data,
         (int)opName.length, opName.data);

  if (!mlirBytecodeIsSentinel(opState->attrDict)) {
    MlirMutableBytesRef attrDictVal = getAttribute(opState->attrDict);
    printf(" %.*s ", (int)attrDictVal.length, attrDictVal.data);
  }

  MlirBytecodeOpHandle op;
  first = true;
  while (
      mlirBytecodeSucceeded(mlirBytecodeReadHandle(&opState->operands, &op))) {
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
      mlirBytecodeReadHandle(&opState->resultTypes, &retTy))) {
    if (first)
      printf(" : ");
    else
      printf(", ");

    if (types && types[retTy.id].length)
      printf("%.*s", (int)types[retTy.id].length, types[retTy.id].data);
    else
      printf("type_%" PRIu64, (uint64_t)retTy.id);

    first = false;
  }

  printf("\n");
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationStateAddRegions(void *callerState, uint64_t n,
                                     MlirBytecodeOperationState *opState) {
  opState->hasRegions = true;
  return printOperationPrefix(callerState, opState);
}

MlirBytecodeStatus
mlirBytecodeOperationStateAddSuccessors(void *callerState,
                                        MlirBytecodeStream *stream, uint64_t numSuccessors,
                                        MlirBytecodeOperationState *opState) {
  printf("Num successors = %d\n", numSuccessors);
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeOperationStateBlockPush(
    void *callerState, MlirBytecodeStream *stream, uint64_t numBlocks,
    MlirBytecodeOperationState *opState) {
  // NOP in printing as in order.
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationStatePop(void *callerState,
                              MlirBytecodeOperationState *opState) {
  if (!opState->hasRegions)
    return printOperationPrefix(callerState, opState);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeIsolatedOperationExit(void *state,
                                                     bool isIsolated) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus printResourceDialect(
    void *state, MlirBytecodeStringHandle groupKey,
    MlirBytecodeSize totalGroups, MlirBytecodeStringHandle resourceKey,
    MlirBytecodeAsmResourceEntryKind kind, const MlirBytecodeBytesRef *blob,
    const uint8_t *boolResource, const MlirBytecodeStringHandle *strHandle) {
  switch (kind) {
  case kMlirBytecodeResourceEntryBlob:
    printf("\t\tblob size(resource %s.%s) = %ld\n",
           mlirBytecodeGetStringSectionValue(state, refFile, groupKey).data,
           mlirBytecodeGetStringSectionValue(state, refFile, resourceKey).data,
           blob->length);
    break;
  case kMlirBytecodeResourceEntryBool:
    printf("\t\tbool resource %s.%s = %d\n",
           mlirBytecodeGetStringSectionValue(state, refFile, groupKey).data,
           mlirBytecodeGetStringSectionValue(state, refFile, resourceKey).data,
           *boolResource);
    break;
  case kMlirBytecodeResourceEntryString:
    printf("\t\tstring resource %s.%s = %s\n",
           mlirBytecodeGetStringSectionValue(state, refFile, groupKey).data,
           mlirBytecodeGetStringSectionValue(state, refFile, resourceKey).data,
           mlirBytecodeGetStringSectionValue(state, refFile, *strHandle).data);
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
          NULL, refFile, &printUnknownAttrs, &printUnknownTypeDialect))) {
    return mlirBytecodeEmitError("MlirBytecodeFailed to parse attr/type"), 1;
  }

  // FIXME: Should be just lazily parsed. Here just doing effectively a local
  // fixed point.
  fprintf(stderr, "Re-parsing attributes & types\n");
  for (int i = 0; i < 3; ++i) {
    fprintf(stderr, "\tparse %d:\n", i);
    if (mlirBytecodeFailed(mlirBytecodeForEachAttributeAndType(
            NULL, refFile, &printUnknownAttrs, &printUnknownTypeDialect))) {
      return mlirBytecodeEmitError("MlirBytecodeFailed to parse attr/type"), 1;
    }
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

#if 0
  if (mlirBytecodeFailed(mlirBytecodeParseFile(
          ref, NULL, &printUnknownAttrs, &printUnknownTypeDialect, &printDialect,
          &printOpDialect, &blockEnterFn, &blockExitFn, &opFn, &opExitFn,
          &regionEnterFn, &regionExitFn, &printResourceDialect, &printStrings)))
    return mlirBytecodeEmitError("MlirBytecodeFailed to parse file"), 1;

  if (mlirBytecodeFailed(mlirBytecodeParseFile(
          ref, NULL, &printUnknownAttrs, &printUnknownTypeDialect, &printDialect,
          &printOpDialect, &blockEnterFn, &blockExitFn, &opFn, &opExitFn,
          &regionEnterFn, &regionExitFn, &printResourceDialect, &printStrings)))
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
