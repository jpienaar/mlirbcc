//===- Builder.cpp - Testing bytecode reader ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/SourceMgr.h"
#include <stack>

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

struct ParsingState;

/// This struct represents a dialect entry within the bytecode.
struct BytecodeDialect {
  /// Load the dialect into the provided context if it hasn't been loaded yet.
  /// Returns failure if the dialect couldn't be loaded *and* the provided
  /// context does not allow unregistered dialects. The provided reader is used
  /// for error emission if necessary.
  LogicalResult load(ParsingState *state, MLIRContext *ctx);

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

struct BytecodeAttribute {
  BytecodeAttribute() : range(), value(nullptr) {}
  BytecodeAttribute(MlirBytecodeAttributeOrTypeRange range) : range(std::move(range)), value(nullptr) {}

  MlirBytecodeAttributeOrTypeRange range;
  Attribute value;
};

struct BytecodeType {
  BytecodeType() : range(), value(nullptr) {}
  BytecodeType(MlirBytecodeAttributeOrTypeRange range) : range(std::move(range)), value(nullptr) {}

  MlirBytecodeAttributeOrTypeRange range;
  Type value;
};

/// This struct represents an operation name entry within the bytecode.
struct BytecodeOperationName {
  BytecodeOperationName(BytecodeDialect *dialect, StringRef name)
      : dialect(dialect), name(name) {}

  /// The loaded operation name, or std::nullopt if it hasn't been processed
  /// yet.
  std::optional<OperationName> opName;

  /// The dialect that owns this operation name.
  BytecodeDialect *dialect;

  /// The name of the operation, without the dialect prefix.
  StringRef name;
};

struct ParsingState {
  ParsingState(Location fileLoc,
               FallbackAsmResourceMap *fallbackResourceMap = nullptr)
      : fallbackResourceMap(fallbackResourceMap), fileLoc(fileLoc) {}

  InFlightDiagnostic emitError(const Twine &msg = {}) {
    return ::emitError(fileLoc, msg);
  }

  Attribute attribute(MlirBytecodeAttrHandle handle) {
    uint64_t i = handle.id;
    if (i >= attributes.size())
      return nullptr;
    if (attributes[i].value)
      return attributes[i].value;
    if (!mlirBytecodeSucceeded(mlirBytecodeProcessAttribute(
            this, (MlirBytecodeAttrHandle){.id = i})))
      return nullptr;
    return attributes[i].value;
  }

  FailureOr<Dialect *> dialect(MlirBytecodeDialectHandle handle) {
    if (handle.id >= dialects.size())
      return failure();
    BytecodeDialect &entry = dialects[handle.id];
    if (entry.dialect)
      return *entry.dialect;
    if (failed(entry.load(this, fileLoc->getContext())))
      return failure();
    return *entry.dialect;
  }

  FailureOr<OperationName> opName(MlirBytecodeOpHandle handle) {
    if (handle.id >= opNames.size())
      return failure();
    BytecodeOperationName &entry = opNames[handle.id];
    if (entry.opName)
      return *entry.opName;
    if (failed(entry.dialect->load(this, fileLoc->getContext())))
      return failure();
    entry.opName = {(entry.dialect->name + "." + entry.name).str(),
                    fileLoc.getContext()};
    return *entry.opName;
  }

  FailureOr<StringRef> string(MlirBytecodeStringHandle handle) {
    if (handle.id >= strings.size())
      return failure();
    return strings[handle.id];
  }

  FailureOr<Type> type(MlirBytecodeTypeHandle handle) {
    uint64_t i = handle.id;
    if (i >= types.size())
      return failure();
    if (types[i].value)
      return types[i].value;
    if (!mlirBytecodeSucceeded(
            mlirBytecodeProcessType(this, (MlirBytecodeTypeHandle){.id = i})))
      return failure();
    return types[i].value;
  }

  // These are all public to enable access via plain C functions.
  std::vector<AsmDialectResourceHandle> resources;
  std::vector<BytecodeAttribute> attributes;
  std::vector<BytecodeDialect> dialects;
  std::vector<BytecodeOperationName> opNames;
  std::vector<BytecodeType> types;
  std::vector<StringRef> strings;

  std::stack<OperationState> operationStateStack;

  DenseMap<StringRef, std::unique_ptr<AsmResourceParser>> resourceParsers;
  FallbackAsmResourceMap *fallbackResourceMap;

  Location fileLoc;
};

LogicalResult BytecodeDialect::load(ParsingState *state, MLIRContext *ctx) {
  if (dialect) {
    return success();
  }
  Dialect *loadedDialect = ctx->getOrLoadDialect(name);
  if (!loadedDialect && !ctx->allowsUnregisteredDialects()) {
    return state->emitError("dialect '")
           << name
           << "' is unknown. If this is intended, please call "
              "allowUnregisteredDialects() on the MLIRContext, or use "
              "-allow-unregistered-dialect with the MLIR tool used";
  }
  dialect = loadedDialect;

  // If the dialect was actually loaded, check to see if it has a bytecode
  // interface.
  if (loadedDialect)
    interface = dyn_cast<BytecodeDialectInterface>(loadedDialect);
  return success();
}

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
  result = state.attribute(handle);
  return success(result);
}

LogicalResult MlirbcDialectBytecodeReader::readType(Type &result) {
  MlirBytecodeTypeHandle handle;
  if (!mlirBytecodeSucceeded(
          mlirBytecodeDialectReaderReadType(&reader, &handle)))
    return failure();
  FailureOr<Type> type = state.type(handle);
  if (failed(type))
    return failure();
  result = *type;
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
mlirBytecodeOperationStatePush(void *callerState, MlirBytecodeOpHandle opHandle,
                               MlirBytecodeLocHandle locHandle,
                               MlirBytecodeOperationStateHandle *opState) {
  ParsingState *state = (ParsingState *)callerState;
  mlirBytecodeEmitDebug("operation state push");
  LocationAttr locAttr =
      dyn_cast_if_present<LocationAttr>(state->attribute(locHandle));
  if (!locAttr)
    return mlirBytecodeEmitError(callerState, "invalid operation location");
  FailureOr<OperationName> opName = state->opName(opHandle);
  if (failed(opName))
    return mlirBytecodeFailure();
  opState->state = &state->operationStateStack.emplace(locAttr, *opName);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationStatePop(void *callerState,
                              MlirBytecodeOperationStateHandle handle) {
  ParsingState *state = (ParsingState *)callerState;
  OperationState *opState = (OperationState *)handle.state;
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddAttributeDictionary(
    void *callerState, MlirBytecodeOperationStateHandle opHandle,
    MlirBytecodeAttrHandle dictHandle) {
  ParsingState *state = (ParsingState *)callerState;
  if (dictHandle.id >= state->attributes.size())
    return mlirBytecodeEmitError(callerState, "out of range attribute handle");
  OperationState *opState = (OperationState *)opHandle.state;
  DictionaryAttr attr =
      dyn_cast_if_present<DictionaryAttr>(state->attribute(dictHandle));
  if (!attr)
    return mlirBytecodeEmitError(callerState, "invalid dictionary attribute");
  opState->addAttributes(attr.getValue());
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddResultTypes(
    void *callerState, MlirBytecodeOperationStateHandle opHandle,
    MlirBytecodeHandlesRef types) {
  ParsingState *state = (ParsingState *)callerState;
  OperationState *opState = (OperationState *)opHandle.state;

  SmallVector<Type> resultTypes;
  resultTypes.reserve(types.length);
  for (uint64_t i = 0, e = types.length; i < e; ++i) {
    MlirBytecodeAttrHandle typeHandle = types.handles[i];
    FailureOr<Type> resultType = state->type(typeHandle);
    if (failed(resultType))
      return mlirBytecodeEmitError(callerState, "invalid result type");
    resultTypes.push_back(*resultType);
  }
  opState->addTypes(resultTypes);

  return mlirBytecodeSuccess();
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
  if (handle.id >= state->attributes.size())
    return mlirBytecodeEmitError(callerState,
                                 "invalid attribute id %" PRIu64 " / %" PRIu64,
                                 handle.id, state->attributes.size());
  BytecodeAttribute& attr = state->attributes[handle.id];
  if (attr.value)
    return mlirBytecodeSuccess();

  if (attr.range.dialectHandle.id >= state->dialects.size())
    return mlirBytecodeEmitError(callerState,
                                 "invalid dialect id %" PRIu64 " / %" PRIu64,
                                 attr.range.dialectHandle.id, state->dialects.size());

  auto &dialect = state->dialects[attr.range.dialectHandle.id];
  if (attr.range.hasCustom) {
    if (failed(dialect.load(state, state->fileLoc.getContext())))
      return mlirBytecodeFailure();

    // Ensure that the dialect implements the bytecode interface.
    if (!dialect.interface) {
      return mlirBytecodeEmitError(
          callerState, "dialect '%s' does not implement the bytecode interface",
          dialect.name.str().c_str());
    }

    // Ask the dialect to parse the entry.
    MlirBytecodeStream stream = mlirBytecodeStreamCreate(attr.range.bytes);
    MlirbcDialectBytecodeReader dialectReader(*state, stream);
    attr.value =
        dialect.interface->readAttribute(dialectReader);
    if (!attr.value)
      return mlirBytecodeFailure();
    return mlirBytecodeSuccess();
  }
    auto asmStr = StringRef((const char *)attr.range.bytes.data, attr.range.bytes.length);
    // Invoke the MLIR assembly parser to parse the entry text.
    size_t numRead = 0;
    MLIRContext *context = state->fileLoc->getContext();
    attr.value
    = ::parseAttribute(asmStr, context, numRead);
    // Ensure there weren't dangling characters after the entry.
    if (numRead != asmStr.size()) {
      return mlirBytecodeEmitError(
          callerState,
          "trailing characters found after Attribute assembly format: %s",
          asmStr.drop_front(numRead).str().c_str());
    }
    if (!attr.value)
      return mlirBytecodeFailure();
    return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeProcessType(void *callerState,
                                           MlirBytecodeTypeHandle handle) {
  ParsingState *state = (ParsingState *)callerState;
  if (handle.id >= state->types.size())
    return mlirBytecodeEmitError(callerState,
                                 "invalid type id %" PRIu64 " / %" PRIu64,
                                 handle.id, state->types.size());
  BytecodeType& type = state->types[handle.id];
  if (type.value)
    return mlirBytecodeSuccess();

  if (type.range.dialectHandle.id >= state->dialects.size())
    return mlirBytecodeEmitError(callerState,
                                 "invalid dialect id %" PRIu64 " / %" PRIu64,
                                 type.range.dialectHandle.id, state->dialects.size());

  auto &dialect = state->dialects[type.range.dialectHandle.id];
  if (type.range.hasCustom) {
    if (failed(dialect.load(state, state->fileLoc.getContext())))
      return mlirBytecodeFailure();

    // Ensure that the dialect implements the bytecode interface.
    if (!dialect.interface) {
      return mlirBytecodeEmitError(
          callerState, "dialect '%s' does not implement the bytecode interface",
          dialect.name.str().c_str());
    }

    // Ask the dialect to parse the entry.
    MlirBytecodeStream stream = mlirBytecodeStreamCreate(type.range.bytes);
    MlirbcDialectBytecodeReader dialectReader(*state, stream);
    type.value = dialect.interface->readType(dialectReader);
    if (!type.value)
      return mlirBytecodeFailure();
    return mlirBytecodeSuccess();
  }

    auto asmStr = StringRef((const char *)type.range.bytes.data, type.range.bytes.length);
    // Invoke the MLIR assembly parser to parse the entry text.
    size_t numRead = 0;
    MLIRContext *context = state->fileLoc->getContext();
    type.value = ::parseType(asmStr, context, numRead);
    // Ensure there weren't dangling characters after the entry.
    if (numRead != asmStr.size()) {
      return mlirBytecodeEmitError(
          callerState,
          "trailing characters found after Type assembly format: %s",
          asmStr.drop_front(numRead).str().c_str());
    }
    if (!type.value)
      return mlirBytecodeFailure();
    return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeAttrCallBack(void *callerState, MlirBytecodeAttrHandle attrHandle,
                         MlirBytecodeDialectHandle dialectHandle,
                         MlirBytecodeBytesRef bytes, bool hasCustom) {
  ParsingState *state = (ParsingState *)callerState;
  state->attributes[attrHandle.id].range = {.bytes = bytes,
  .hasCustom = hasCustom,
  .dialectHandle = dialectHandle};
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeTypeCallBack(void *callerState, MlirBytecodeTypeHandle typeHandle,
                         MlirBytecodeDialectHandle dialectHandle,
                         MlirBytecodeBytesRef bytes, bool hasCustom) {
  ParsingState *state = (ParsingState *)callerState;
  state->types[typeHandle.id].range = {bytes = bytes,
  .hasCustom = hasCustom,
  .dialectHandle = dialectHandle};
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeDialectCallBack(void *callerState,
                            MlirBytecodeDialectHandle dialectHandle,
                            MlirBytecodeStringHandle stringHandle) {
  ParsingState *state = (ParsingState *)callerState;
  if (dialectHandle.id >= state->dialects.size())
    return mlirBytecodeFailure();
  BytecodeDialect &dialect = state->dialects[dialectHandle.id];
  auto name = state->string(stringHandle);
  if (failed(name))
    return mlirBytecodeFailure();
  dialect.name = *name;
  mlirBytecodeEmitDebug("dialect[%d] = %s", (int)dialectHandle.id, *name);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeDialectOpCallBack(void *callerState, MlirBytecodeOpHandle opHandle,
                              MlirBytecodeDialectHandle dialectHandle,
                              MlirBytecodeStringHandle strHandle) {
  ParsingState *state = (ParsingState *)callerState;
  assert(state->opNames.size() == opHandle.id);

  if (dialectHandle.id >= state->dialects.size())
    return mlirBytecodeEmitError(callerState, "invalid dialect");
  BytecodeDialect *dialect = &state->dialects[dialectHandle.id];
  FailureOr<StringRef> name = state->string(strHandle);
  if (failed(name))
    return mlirBytecodeEmitError(callerState, "invalid op name");
  state->opNames.emplace_back(dialect, *name);
  return mlirBytecodeSuccess();
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
                                  MlirBytecodeStringHandle strHandle,
                                  MlirBytecodeBytesRef *result) {
  ParsingState *state = (ParsingState *)callerState;
  auto str = state->string(strHandle);
  if (failed(str))
    return mlirBytecodeEmitError(callerState, "invalid string reference");
  result->data = (const uint8_t *)str->data();
  result->length = str->size();
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeAttributesPush(void *callerState,
                                              MlirBytecodeSize numArgs) {
  ParsingState *state = (ParsingState *)callerState;
  state->attributes.resize(numArgs);
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
