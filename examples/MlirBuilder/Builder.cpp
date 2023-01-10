//===- Builder.cpp - Testing bytecode reader ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/Diagnostics.h"

// Include bytecode parsing implementation.
#include "mlirbcc/BytecodeTypes.h"
#include "mlirbcc/DialectBytecodeReader.c.inc"
#include "mlirbcc/Parse.c.inc"

using namespace mlir;

struct MlirbcDialectBytecodeReader : public mlir::DialectBytecodeReader {
  MlirbcDialectBytecodeReader(MlirBytecodeDialectReader reader, Location loc)
      : loc(loc){};

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

  Attribute attribute(int i) const { return attributes[i]; }
  Type type(int i) const { return types[i]; }

  MlirbcDialectBytecodeReader &reader;

  // These are all public to enable access via plain C functions.
  std::vector<Attribute> attributes;
  std::vector<Type> types;
  std::vector<AsmDialectResourceHandle> resources;
  std::vector<StringRef> strings;

  Location loc;
};

InFlightDiagnostic MlirbcDialectBytecodeReader::emitError(const Twine &msg) {
  return mlir::emitError(loc, msg);
}

LogicalResult MlirbcDialectBytecodeReader::readAttribute(Attribute &result) {
  MlirBytecodeAttrHandle handle;
  if (!mlirBytecodeSucceeded(
          mlirBytecodeDialectReaderReadAttribute(&reader, handle)))
    return failure();
  result = attribute(handle.id);
  return success();
}

LogicalResult MlirbcDialectBytecodeReader::readType(Type &result) {
  MlirBytecodeTypeHandle handle;
  if (!mlirBytecodeSucceeded(
          mlirBytecodeDialectReaderReadType(&reader, handle)))
    return failure();
  result = type(handle.id);
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
  MlirBytecodeStringHandle handle;
  if (!mlirBytecodeSucceeded(mlirBytecodeParseHandle(this, &stream, &handle)))
    return failure();
  MlirBytecodeBytesRef ref;
  if (!mlirBytecodeSucceeded(
          mlirBytecodeDialectReaderReadString(&reader, &ref)))
    return failure();
  result = StringRef(ref.data, ref.length);
  return success();
}

LogicalResult MlirbcDialectBytecodeReader::readBlob(ArrayRef<char> &result) {
  MlirBytecodeBytesRef ref;
  if (!mlirBytecodeSucceeded(mlirBytecodeDialectReaderReadBlob(&reader, &ref)))
    return failure();
  result = ArrayRef(ref.data, ref.length);
  return success();
}

FailureOr<AsmDialectResourceHandle>
MlirbcDialectBytecodeReader::readResourceHandle() {
  MlirBytecodeResourceHandle handle;
  if (!mlirBytecodeSucceeded(mlirBytecodeParseHandle(this, &stream, &handle)))
    return failure();
  if (handle.id >= resources.size())
    return failure();
  return resources[handle.id];
}

MlirBytecodeStatus
mlirBytecodeOperationStatePush(void *callerState, MlirBytecodeOpHandle name,
                               MlirBytecodeLocHandle loc,
                               MlirBytecodeOperationStateHandle *opState) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeOperationStatePop(void *callerState,
                              MlirBytecodeOperationStateHandle) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddAttributeDictionary(
    void *callerState, MlirBytecodeOperationStateHandle,
    MlirBytecodeAttrHandle) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeOperationStateAddResultTypes(void *callerState,
                                         MlirBytecodeOperationStateHandle,
                                         MlirBytecodeHandlesRef types) {
  return mlirBytecodeUnhandled();
}

// Note: This _requires_ that the stream is progressed post the last item before
// this function returns.
MlirBytecodeStatus mlirBytecodeOperationStateAddOperands(
    void *callerState, MlirBytecodeOperationStateHandle, MlirBytecodeStream *,
    uint64_t numOperands) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddRegions(
    void *callerState, MlirBytecodeOperationStateHandle, uint64_t numRegions) {
  return mlirBytecodeUnhandled();
}

// Note: This _requires_ that the stream is progressed post the last item before
// this function returns.
MlirBytecodeStatus mlirBytecodeOperationStateAddSuccessors(
    void *callerState, MlirBytecodeOperationStateHandle, MlirBytecodeStream *,
    uint64_t numSuccessors) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeRegionPush(void *callerState,
                                          MlirBytecodeOperationStateHandle,
                                          size_t numBlocks, size_t numValues) {
  return mlirBytecodeUnhandled();
}

// Note: This _requires_ that the stream is progressed post the last item before
// this function returns.
MlirBytecodeStatus
mlirBytecodeOperationStateBlockPush(void *callerState,
                                    MlirBytecodeOperationStateHandle,
                                    MlirBytecodeStream *, uint64_t numArgs) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeOperationStateBlockPop(void *callerState,
                                   MlirBytecodeOperationStateHandle) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeOperationStateRegionPop(void *callerState,
                                    MlirBytecodeOperationStateHandle) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeProcessAttribute(void *callerState,
                                                MlirBytecodeAttrHandle handle) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeProcessType(void *callerState,
                                           MlirBytecodeTypeHandle handle) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeAttrCallBack(void *, MlirBytecodeAttrHandle,
                                            size_t total,
                                            MlirBytecodeDialectHandle,
                                            MlirBytecodeBytesRef,
                                            bool hasCustom) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeTypeCallBack(void *, MlirBytecodeTypeHandle,
                                            size_t total,
                                            MlirBytecodeDialectHandle,
                                            MlirBytecodeBytesRef,
                                            bool hasCustom) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeDialectCallBack(void *,
                                               MlirBytecodeDialectHandle,
                                               size_t total,
                                               MlirBytecodeStringHandle) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeDialectOpCallBack(void *,
                                                 MlirBytecodeDialectHandle,
                                                 MlirBytecodeOpHandle,
                                                 MlirBytecodeStringHandle) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeResourceSectionEnter(void *,
                                 MlirBytecodeSize numExternalResourceGroups) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeResourceGroupEnter(void *, MlirBytecodeStringHandle groupKey,
                               MlirBytecodeSize numResources) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeResourceBlobCallBack(void *, MlirBytecodeStringHandle resourceKey,
                                 MlirBytecodeStringHandle groupKey,
                                 MlirBytecodeBytesRef blob) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeResourceBoolCallBack(void *, MlirBytecodeStringHandle resourceKey,
                                 MlirBytecodeStringHandle groupKey,
                                 const uint8_t) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeResourceStringCallBack(void *, MlirBytecodeStringHandle resourceKey,
                                   MlirBytecodeStringHandle groupKey,
                                   MlirBytecodeStringHandle) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeBytesRef
mlirBytecodeGetStringSectionValue(void *callerState,
                                  MlirBytecodeStringHandle idx) {
  auto *state = (MlirbcDialectBytecodeReader *)callerState;
  auto str = state->strings[idx.id];
  return MlirBytecodeBytesRef{.data = (const uint8_t *)str.data(),
                              .length = str.size()};
}

MlirBytecodeStatus mlirBytecodeStringCallBack(void *, MlirBytecodeStringHandle,
                                              size_t total,
                                              MlirBytecodeBytesRef) {
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
  MlirBytecodeDialectReader reader = {
      .callerState = this,
      .stream = (MlirBytecodeStream){
          .start = ref.data, .end = ref.data, .pos = ref.data}};

  MlirbcDialectBytecodeReader reader(loc);
  MlirBytecodeParserState parserState =
      mlirBytecodePopulateParserState(&reader, ref);
  if (!mlirBytecodeParserStateEmpty(&parserState)) {
    if (mlirBytecodeFailed(mlirBytecodeParse(&reader, ref)))
      return mlirBytecodeEmitError(&reader, "MlirBytecodeFailed to parse file"),
             1;
  }

  return 0;
}
