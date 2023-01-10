//===- Builder.cpp - Testing bytecode reader ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/SourceMgr.h"

// Include bytecode parsing implementation.
#include "mlirbcc/BytecodeTypes.h"
#include "mlirbcc/Parse.c.inc"
// Dialect and attribute parsing helpers.
#include "mlirbcc/DialectBytecodeReader.c.inc"

using namespace mlir;

// TODO: Support for big-endian architectures.
// TODO: Properly preserve use lists of values.

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/SourceMgr.h"
#include <optional>

#define DEBUG_TYPE "mlir-bytecode-reader"

struct ParsingState {
  ParsingState(Location loc) : loc(loc) {}

  Attribute attribute(int i) const { return attributes[i]; }
  Type type(int i) const { return types[i]; }
  StringRef string(int i) const { return strings[i]; }

  // These are all public to enable access via plain C functions.
  std::vector<Attribute> attributes;
  std::vector<Type> types;
  std::vector<AsmDialectResourceHandle> resources;
  std::vector<StringRef> strings;

  Location loc;
};

struct MlirbcDialectBytecodeReader : public mlir::DialectBytecodeReader {
  MlirbcDialectBytecodeReader(ParsingState &state, MlirBytecodeStream &stream)
      : reader((MlirBytecodeDialectReader){.callerState = this,
                                           .stream = &stream}),
        state(state){};

  InFlightDiagnostic emitError(const Twine &msg = {}) override;
  LogicalResult readAttribute(Attribute &result) override;
  LogicalResult readType(Type &result) override;
  LogicalResult readVarInt(uint64_t &result) override;
  LogicalResult readSignedVarInt(int64_t &result) override;
  FailureOr<APInt> readAPIntWithKnownWidth(unsigned bitWidth) override;
  FailureOr<APFloat>
  readAPFloatWithKnownSemantics(const llvm::fltSemantics &semantics) override;
  LogicalResult readString(StringRef &result) override;
  LogicalResult readBlob(ArrayRef<char> &result) override;
  FailureOr<AsmDialectResourceHandle> readResourceHandle() override;

  MlirBytecodeDialectReader reader;
  ParsingState &state;
};

InFlightDiagnostic MlirbcDialectBytecodeReader::emitError(const Twine &msg) {
  return mlir::emitError(state.loc, msg);
}

LogicalResult MlirbcDialectBytecodeReader::readAttribute(Attribute &result) {
  MlirBytecodeAttrHandle handle;
  if (!mlirBytecodeSucceeded(
          mlirBytecodeDialectReaderReadAttribute(&reader, &handle)))
    return failure();
  result = state.attribute(handle.id);
  return success();
}

LogicalResult MlirbcDialectBytecodeReader::readType(Type &result) {
  MlirBytecodeTypeHandle handle;
  if (!mlirBytecodeSucceeded(
          mlirBytecodeDialectReaderReadType(&reader, &handle)))
    return failure();
  result = state.type(handle.id);
  return success();
}

LogicalResult MlirbcDialectBytecodeReader::readVarInt(uint64_t &result) {
  return failure(!mlirBytecodeSucceeded(
      mlirBytecodeDialectReaderReadVarInt(&reader, &result)));
}

LogicalResult MlirbcDialectBytecodeReader::readSignedVarInt(int64_t &result) {
  return failure(!mlirBytecodeSucceeded(
      mlirBytecodeDialectReaderReadSignedVarInt(&reader, &result)));
}

FailureOr<APInt>
MlirbcDialectBytecodeReader::readAPIntWithKnownWidth(unsigned bitWidth) {

  MlirBytecodeAPInt result;
  MlirBytecodeStatus ret =
      mlirBytecodeParseAPIntWithKnownWidth(&reader, bitWidth, malloc, &result);
  if (!mlirBytecodeSucceeded(ret))
    return failure();
  if (result.bitWidth <= 64)
    return APInt(result.bitWidth, result.U.value);

  const uint64_t bitsPerWord = sizeof(uint64_t) * CHAR_BIT;
  uint64_t numWords = ((uint64_t)bitWidth + bitsPerWord - 1) / bitsPerWord;
  APInt retVal(bitWidth, llvm::makeArrayRef(result.U.data, numWords));
  return retVal;
}

FailureOr<APFloat> MlirbcDialectBytecodeReader::readAPFloatWithKnownSemantics(
    const llvm::fltSemantics &semantics) {
  FailureOr<APInt> intVal =
      readAPIntWithKnownWidth(APFloat::getSizeInBits(semantics));
  if (failed(intVal))
    return failure();
  return APFloat(semantics, *intVal);
  return failure();
}

LogicalResult MlirbcDialectBytecodeReader::readString(StringRef &result) {
  MlirBytecodeBytesRef ref;
  if (!mlirBytecodeSucceeded(
          mlirBytecodeDialectReaderReadString(&reader, &ref)))
    return failure();
  result = StringRef((const char *)ref.data, ref.length);
  return success();
}

LogicalResult MlirbcDialectBytecodeReader::readBlob(ArrayRef<char> &result) {
  MlirBytecodeBytesRef ref;
  if (!mlirBytecodeSucceeded(mlirBytecodeDialectReaderReadBlob(&reader, &ref)))
    return failure();
  result = ArrayRef((const char *)ref.data, ref.length);
  return success();
}

FailureOr<AsmDialectResourceHandle>
MlirbcDialectBytecodeReader::readResourceHandle() {
  MlirBytecodeResourceHandle handle;
  if (!mlirBytecodeSucceeded(
          mlirBytecodeDialectReaderReadResourceHandle(&reader, &handle)))
    return failure();
  if (handle.id >= state.resources.size())
    return failure();
  return state.resources[handle.id];
}

MlirBytecodeStatus
mlirBytecodeOperationStatePush(void *callerState, MlirBytecodeOpHandle name,
                               MlirBytecodeLocHandle loc,
                               MlirBytecodeOperationStateHandle *opState) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeOperationStatePop(void *callerState,
                              MlirBytecodeOperationStateHandle) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddAttributeDictionary(
    void *callerState, MlirBytecodeOperationStateHandle,
    MlirBytecodeAttrHandle) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeOperationStateAddResultTypes(void *callerState,
                                         MlirBytecodeOperationStateHandle,
                                         MlirBytecodeHandlesRef types) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

// Note: This _requires_ that the stream is progressed post the last item before
// this function returns.
MlirBytecodeStatus
mlirBytecodeOperationStateAddOperands(void *callerState,
                                      MlirBytecodeOperationStateHandle,
                                      MlirBytecodeHandlesRef) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddRegions(
    void *callerState, MlirBytecodeOperationStateHandle, uint64_t numRegions) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

// Note: This _requires_ that the stream is progressed post the last item before
// this function returns.
MlirBytecodeStatus mlirBytecodeOperationStateAddSuccessors(
    void *callerState, MlirBytecodeOperationStateHandle, MlirBytecodeSizesRef) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeRegionPush(void *callerState,
                                          MlirBytecodeOperationStateHandle,
                                          size_t numBlocks, size_t numValues) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

// Note: This _requires_ that the stream is progressed post the last item before
// this function returns.
MlirBytecodeStatus
mlirBytecodeOperationStateBlockPush(void *callerState,
                                    MlirBytecodeOperationStateHandle,
                                    MlirBytecodeHandlesRef) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeOperationStateBlockPop(void *callerState,
                                   MlirBytecodeOperationStateHandle) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeOperationStateRegionPop(void *callerState,
                                    MlirBytecodeOperationStateHandle) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeProcessAttribute(void *callerState,
                                                MlirBytecodeAttrHandle handle) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeProcessType(void *callerState,
                                           MlirBytecodeTypeHandle handle) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeAttrCallBack(void *callerState,
                                            MlirBytecodeAttrHandle,
                                            MlirBytecodeDialectHandle,
                                            MlirBytecodeBytesRef,
                                            bool hasCustom) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeTypeCallBack(void *callerState,
                                            MlirBytecodeTypeHandle,
                                            MlirBytecodeDialectHandle,
                                            MlirBytecodeBytesRef,
                                            bool hasCustom) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeDialectCallBack(void *callerState,
                                               MlirBytecodeDialectHandle,
                                               MlirBytecodeStringHandle) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeDialectOpCallBack(void *callerState,
                                                 MlirBytecodeDialectHandle,
                                                 MlirBytecodeOpHandle,
                                                 MlirBytecodeStringHandle) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeResourceSectionEnter(void *callerState,
                                 MlirBytecodeSize numExternalResourceGroups) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeResourceGroupEnter(void *callerState,
                               MlirBytecodeStringHandle groupKey,
                               MlirBytecodeSize numResources) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeResourceBlobCallBack(
    void *callerState, MlirBytecodeStringHandle resourceKey,
    MlirBytecodeStringHandle groupKey, MlirBytecodeBytesRef blob) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeResourceBoolCallBack(
    void *callerState, MlirBytecodeStringHandle resourceKey,
    MlirBytecodeStringHandle groupKey, const uint8_t) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeResourceStringCallBack(
    void *callerState, MlirBytecodeStringHandle resourceKey,
    MlirBytecodeStringHandle groupKey, MlirBytecodeStringHandle) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeGetStringSectionValue(void *callerState,
                                  MlirBytecodeStringHandle idx,
                                  MlirBytecodeBytesRef *result) {
  ParsingState *state = (ParsingState *)callerState;
  auto str = state->string(idx.id);
  result->data = (const uint8_t *)str.data();
  result->length = str.size();
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeAttributesPush(void *callerState,
                                              MlirBytecodeSize numArgs) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeDialectsPush(void *callerState,
                                            MlirBytecodeSize numDialects) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeStringsPush(void *callerState,
                                           MlirBytecodeSize numStrings) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeTypesPush(void *callerState,
                                         MlirBytecodeSize numTypes) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeStringCallBack(void *callerState,
                                              MlirBytecodeStringHandle,
                                              MlirBytecodeBytesRef) {
  ParsingState *state = (ParsingState *)callerState;
  return mlirBytecodeUnhandled();
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s file.mlirbc\n", argv[0]);
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(argv[1]);
  if (std::error_code error = fileOrErr.getError()) {
    fprintf(stderr, "MlirBytecodeFailed to open file '%s'", argv[1]);
    return 1;
  }
  auto idx = sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  const llvm::MemoryBuffer *buffer = sourceMgr.getMemoryBuffer(idx);

  MlirBytecodeBytesRef ref = {.data = (const uint8_t *)buffer->getBufferStart(),
                              .length = buffer->getBufferSize()};
  MLIRContext context;
  auto loc = UnknownLoc::get(&context);
  MlirBytecodeStream stream = {
      .start = ref.data, .pos = ref.data, .end = ref.data};

  MlirBytecodeParserState parserState =
      mlirBytecodePopulateParserState(&stream, ref, nullptr, 0);
  ParsingState state(loc);
  if (!mlirBytecodeParserStateEmpty(&parserState)) {
    if (mlirBytecodeFailed(mlirBytecodeParse(&state, &parserState)))
      return mlirBytecodeEmitError(&state, "MlirBytecodeFailed to parse file"),
             1;
  }

  return 0;
}
