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
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/SourceMgr.h"
#include <optional>

#define DEBUG_TYPE "mlir-bytecode-reader"

//===----------------------------------------------------------------------===//
// BytecodeDialect
//===----------------------------------------------------------------------===//

namespace {

struct BytecodeDialect {
  /// Load the dialect into the provided context if it hasn't been loaded yet.
  /// Returns failure if the dialect couldn't be loaded *and* the provided
  /// context does not allow unregistered dialects. The provided reader is used
  /// for error emission if necessary.
  LogicalResult load(StringRef name, MLIRContext *ctx) {
    if (dialect) {
      return success();
    }
    Dialect *loadedDialect = ctx->getOrLoadDialect(name);
    if (!loadedDialect && !ctx->allowsUnregisteredDialects()) {
      return failure();
      /*
      reader.emitError(
          "dialect '", name,
          "' is unknown. If this is intended, please call "
          "allowUnregisteredDialects() on the MLIRContext, or use "
          "-allow-unregistered-dialect with the MLIR tool used.");
       * */
    }
    dialect = loadedDialect;

    // If the dialect was actually loaded, check to see if it has a bytecode
    // interface.
    if (loadedDialect)
      interface = dyn_cast<BytecodeDialectInterface>(loadedDialect);
    return success();
  }

  /// Return the loaded dialect, or nullptr if the dialect is unknown. This can
  /// only be called after `load`.
  Dialect *getLoadedDialect() const {
    assert(dialect &&
           "expected `load` to be invoked before `getLoadedDialect`");
    return *dialect;
  }

  /// The loaded dialect entry. This field is std::nullopt if we haven't
  /// attempted to load, nullptr if we failed to load, otherwise the loaded
  /// dialect.
  std::optional<Dialect *> dialect;

  /// The bytecode interface of the dialect, or nullptr if the dialect does not
  /// implement the bytecode interface. This field should only be checked if the
  /// `dialect` field is not std::nullopt.
  const BytecodeDialectInterface *interface = nullptr;

  /// The name of the dialect.
  StringRef name;
};

struct MlirBytecodeAttributeOrTypeRange {
  MlirBytecodeBytesRef bytes;
  MlirBytecodeDialectHandle dialectHandle;
  bool hasCustom;
};

struct ParsingState {
  ParsingState(Location fileLoc,
               FallbackAsmResourceMap *fallbackResourceMap = nullptr)
      : fallbackResourceMap(fallbackResourceMap), fileLoc(fileLoc) {}

  InFlightDiagnostic emitError(const Twine &msg = {}) {
    llvm::errs() << "foo: " << msg << "\n";
    return ::emitError(fileLoc, msg);
  }

  Attribute attribute(int i) const { return attributes[i]; }
  Type type(int i) const { return types[i]; }
  StringRef string(int i) const { return strings[i]; }

  // Mapping from attribute handle to range.
  std::vector<MlirBytecodeAttributeOrTypeRange> attributeRange;
  // Mapping from type handle to range.
  // TODO: Should this be merged with attributes and lazily change to processed
  // form?
  std::vector<MlirBytecodeAttributeOrTypeRange> typeRange;

  // These are all public to enable access via plain C functions.
  std::vector<AsmDialectResourceHandle> resources;
  std::vector<Attribute> attributes;
  std::vector<BytecodeDialect> dialects;
  std::vector<StringRef> strings;
  std::vector<Type> types;

  DenseMap<StringRef, std::unique_ptr<AsmResourceParser>> resourceParsers;
  FallbackAsmResourceMap *fallbackResourceMap;

  Location fileLoc;
};

} // namespace

struct MlirbcDialectBytecodeReader : public mlir::DialectBytecodeReader {
  MlirbcDialectBytecodeReader(ParsingState &state, MlirBytecodeStream &stream)
      : reader((MlirBytecodeDialectReader){.callerState = &state,
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
  return state.emitError(msg);
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
                               MlirBytecodeLocHandle fileLoc,
                               MlirBytecodeOperationStateHandle *opState) {
  ParsingState *state = (ParsingState *)callerState;
  mlirBytecodeEmitDebug("operation state push");
  for (auto it : llvm::enumerate(state->typeRange)) {
    llvm::errs() << "With " << it.index() << ": ";
    if (!mlirBytecodeSucceeded(mlirBytecodeProcessType(
            callerState, (MlirBytecodeTypeHandle){.id = it.index()})))
      return mlirBytecodeFailure();
    llvm::errs() << "\n";
  }
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
  mlirBytecodeEmitDebug("processing attribute %d", (int)handle.id);

  ParsingState *state = (ParsingState *)callerState;
  MlirBytecodeAttributeOrTypeRange attr = state->attributeRange[handle.id];
  if (handle.id >= state->attributes.size())
    return mlirBytecodeEmitError(callerState,
                                 "invalid attribute id %" PRIu64 " / %" PRIu64,
                                 handle.id, state->attributes.size());
  if (attr.dialectHandle.id >= state->dialects.size())
    return mlirBytecodeEmitError(callerState,
                                 "invalid dialect id %" PRIu64 " / %" PRIu64,
                                 attr.dialectHandle.id, state->dialects.size());

  auto &dialect = state->dialects[attr.dialectHandle.id];
  if (attr.hasCustom) {
    if (failed(dialect.load(dialect.name, state->fileLoc.getContext())))
      return mlirBytecodeFailure();

    // Ensure that the dialect implements the bytecode interface.
    if (!dialect.interface) {
      return mlirBytecodeEmitError(
          callerState, "dialect '%s' does not implement the bytecode interface",
          dialect.name.str().c_str());
    }

    // Ask the dialect to parse the entry.
    MlirBytecodeStream stream = mlirBytecodeStreamCreate(attr.bytes);
    MlirbcDialectBytecodeReader dialectReader(*state, stream);
    state->attributes[handle.id] =
        dialect.interface->readAttribute(dialectReader);
    if (!state->attributes[handle.id])
      return mlirBytecodeFailure();
    state->attributes[handle.id].dump();
    return mlirBytecodeSuccess();
  } else {
    auto asmStr = StringRef((const char *)attr.bytes.data, attr.bytes.length);
    // Invoke the MLIR assembly parser to parse the entry text.
    size_t numRead = 0;
    MLIRContext *context = state->fileLoc->getContext();
    state->attributes[handle.id] = ::parseAttribute(asmStr, context, numRead);
    // Ensure there weren't dangling characters after the entry.
    if (numRead != asmStr.size()) {
      return mlirBytecodeEmitError(
          callerState,
          "trailing characters found after Attribute assembly format: %s",
          asmStr.drop_front(numRead).str().c_str());
    }
    if (!state->attributes[handle.id])
      return mlirBytecodeFailure();
    state->attributes[handle.id].dump();
    return mlirBytecodeSuccess();
  }

  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeProcessType(void *callerState,
                                           MlirBytecodeTypeHandle handle) {
  ParsingState *state = (ParsingState *)callerState;
  MlirBytecodeAttributeOrTypeRange type = state->typeRange[handle.id];
  if (handle.id >= state->types.size())
    return mlirBytecodeEmitError(callerState,
                                 "invalid type id %" PRIu64 " / %" PRIu64,
                                 handle.id, state->types.size());
  if (type.dialectHandle.id >= state->dialects.size())
    return mlirBytecodeEmitError(callerState,
                                 "invalid dialect id %" PRIu64 " / %" PRIu64,
                                 type.dialectHandle.id, state->dialects.size());

  auto &dialect = state->dialects[type.dialectHandle.id];
  if (type.hasCustom) {
    if (failed(dialect.load(dialect.name, state->fileLoc.getContext())))
      return mlirBytecodeFailure();

    // Ensure that the dialect implements the bytecode interface.
    if (!dialect.interface) {
      return mlirBytecodeEmitError(
          callerState, "dialect '%s' does not implement the bytecode interface",
          dialect.name.str().c_str());
    }

    // Ask the dialect to parse the entry.
    MlirBytecodeStream stream = mlirBytecodeStreamCreate(type.bytes);
    MlirbcDialectBytecodeReader dialectReader(*state, stream);
    state->types[handle.id] = dialect.interface->readType(dialectReader);
    if (!state->types[handle.id])
      return mlirBytecodeFailure();
    // TEST
    state->types[handle.id].dump();
    return mlirBytecodeSuccess();
  } else {
    auto asmStr = StringRef((const char *)type.bytes.data, type.bytes.length);
    // Invoke the MLIR assembly parser to parse the entry text.
    size_t numRead = 0;
    MLIRContext *context = state->fileLoc->getContext();
    state->types[handle.id] = ::parseType(asmStr, context, numRead);
    // Ensure there weren't dangling characters after the entry.
    if (numRead != asmStr.size()) {
      return mlirBytecodeEmitError(
          callerState,
          "trailing characters found after Type assembly format: %s",
          asmStr.drop_front(numRead).str().c_str());
    }
    if (!state->types[handle.id])
      return mlirBytecodeFailure();
    // TEST
    state->types[handle.id].dump();
    return mlirBytecodeSuccess();
  }

  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeAttrCallBack(void *callerState, MlirBytecodeAttrHandle attrHandle,
                         MlirBytecodeDialectHandle dialectHandle,
                         MlirBytecodeBytesRef bytes, bool hasCustom) {
  ParsingState *state = (ParsingState *)callerState;
  state->attributeRange[attrHandle.id].bytes = bytes;
  state->attributeRange[attrHandle.id].hasCustom = hasCustom;
  state->attributeRange[attrHandle.id].dialectHandle = dialectHandle;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeTypeCallBack(void *callerState, MlirBytecodeTypeHandle typeHandle,
                         MlirBytecodeDialectHandle dialectHandle,
                         MlirBytecodeBytesRef bytes, bool hasCustom) {
  ParsingState *state = (ParsingState *)callerState;
  state->typeRange[typeHandle.id].bytes = bytes;
  state->typeRange[typeHandle.id].hasCustom = hasCustom;
  state->typeRange[typeHandle.id].dialectHandle = dialectHandle;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeDialectCallBack(void *callerState,
                            MlirBytecodeDialectHandle dialectHandle,
                            MlirBytecodeStringHandle stringHandle) {
  ParsingState *state = (ParsingState *)callerState;
  BytecodeDialect &dialect = state->dialects[dialectHandle.id];
  mlirBytecodeEmitDebug("dialect %d parsing ", (int)dialectHandle.id);
  if (stringHandle.id >= state->strings.size())
    return mlirBytecodeFailure();
  dialect.name = state->string(stringHandle.id);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeDialectOpCallBack(
    void *callerState, MlirBytecodeDialectHandle dialectHandle,
    MlirBytecodeOpHandle opHandle, MlirBytecodeStringHandle strHandle) {
  ParsingState *state = (ParsingState *)callerState;
  // fIXME
  return mlirBytecodeSuccess();
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
  state->attributes.resize(numArgs);
  state->attributeRange.resize(numArgs);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeDialectsPush(void *callerState,
                                            MlirBytecodeSize numDialects) {
  ParsingState *state = (ParsingState *)callerState;
  state->dialects.resize(numDialects);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeStringsPush(void *callerState,
                                           MlirBytecodeSize numStrings) {
  ParsingState *state = (ParsingState *)callerState;
  state->strings.resize(numStrings);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeTypesPush(void *callerState,
                                         MlirBytecodeSize numTypes) {
  ParsingState *state = (ParsingState *)callerState;
  state->types.resize(numTypes);
  state->typeRange.resize(numTypes);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeStringCallBack(void *callerState,
                                              MlirBytecodeStringHandle handle,
                                              MlirBytecodeBytesRef bytes) {
  ParsingState *state = (ParsingState *)callerState;
  if (handle.id >= state->strings.size())
    return mlirBytecodeFailure();

  state->strings[handle.id] = StringRef((const char *)bytes.data, bytes.length);
  return mlirBytecodeUnhandled();
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s file.mlirbc\n", argv[0]);
    return 1;
  }
  std::string fileName(argv[1]);

  llvm::InitLLVM y(argc, argv);
  registerMLIRContextCLOptions();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(fileName);
  if (!fileOrErr) {
    std::error_code error = fileOrErr.getError();
    fprintf(stderr, "MlirBytecodeFailed to open file '%s'", argv[1]);
    return 1;
  }
  llvm::SourceMgr sourceMgr;
  auto idx = sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  const llvm::MemoryBuffer *buffer = sourceMgr.getMemoryBuffer(idx);

  MlirBytecodeBytesRef ref = {.data = (const uint8_t *)buffer->getBufferStart(),
                              .length = buffer->getBufferSize()};
  DialectRegistry registry;
  MLIRContext context(registry);

  auto fileLoc = UnknownLoc::get(&context);
  MlirBytecodeStream stream = {
      .start = ref.data, .pos = ref.data, .end = ref.data};

  MlirBytecodeParserState parserState =
      mlirBytecodePopulateParserState(&stream, ref, nullptr, 0);
  ParsingState state(fileLoc);
  if (!mlirBytecodeParserStateEmpty(&parserState)) {
    if (mlirBytecodeFailed(mlirBytecodeParse(&state, &parserState)))
      return mlirBytecodeEmitError(&state, "MlirBytecodeFailed to parse file"),
             1;
  }

  return 0;
}
