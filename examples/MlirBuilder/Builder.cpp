//===- Builder.cpp - Testing bytecode reader ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
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
#include <stack>

// Parsing inc configuration.
typedef mlir::OperationState MlirBytecodeOperationState;
typedef mlir::Operation MlirBytecodeOperation;
// Include bytecode parsing implementation.
#include "mlirbcc/BytecodeTypes.h"
#include "mlirbcc/Parse.c.inc"
// Dialect and attribute parsing helpers.
#include "mlirbcc/DialectBytecodeReader.c.inc"

#define DEBUG_TYPE "mlir-bytecode-reader"

using namespace mlir;

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

/// Range corresponding to Attribute or Type.
struct MlirBytecodeAttributeOrTypeRange {
  MlirBytecodeBytesRef bytes;
  MlirBytecodeDialectHandle dialectHandle;
  bool hasCustom;
};

/// Represent either range in file or materialized Attribute.
struct BytecodeAttribute {
  BytecodeAttribute() : range(), value(nullptr) {}
  BytecodeAttribute(MlirBytecodeAttributeOrTypeRange range)
      : range(std::move(range)), value(nullptr) {}

  MlirBytecodeAttributeOrTypeRange range;
  Attribute value;
};

/// Represent either range in file or materialized Type.
struct BytecodeType {
  BytecodeType() : range(), value(nullptr) {}
  BytecodeType(MlirBytecodeAttributeOrTypeRange range)
      : range(std::move(range)), value(nullptr) {}

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

/// This struct represents the current read state of a range of regions. This
/// struct is used to enable iterative parsing of regions.
struct RegionReadState {
  RegionReadState(Operation *op, bool isIsolatedFromAbove)
      : RegionReadState(op->getRegions(), isIsolatedFromAbove) {}
  RegionReadState(MutableArrayRef<Region> regions, bool isIsolatedFromAbove)
      : curRegion(regions.begin()), endRegion(regions.end()),
        isIsolatedFromAbove(isIsolatedFromAbove) {}

  /// The current regions being read.
  MutableArrayRef<Region>::iterator curRegion, endRegion;

  /// The number of values defined immediately within this region.
  unsigned numValues = 0;

  /// The current blocks of the region being read.
  SmallVector<Block *> curBlocks;
  SmallVector<Block *>::iterator curBlock = {};

  /// The number of operations remaining to be read from the current block
  /// being read.
  uint64_t numOpsRemaining = 0;

  /// A flag indicating if the regions being read are isolated from above.
  bool isIsolatedFromAbove = false;
};

/// This class represents a single value scope, in which a value scope is
/// delimited by isolated from above regions.
struct ValueScope {
  /// Push a new region state onto this scope, reserving enough values for
  /// those defined within the current region of the provided state.
  void push(RegionReadState &readState) {
    nextValueIDs.push_back(values.size());
    values.resize(values.size() + readState.numValues);
  }

  /// Pop the values defined for the current region within the provided region
  /// state.
  void pop(RegionReadState &readState) {
    values.resize(values.size() - readState.numValues);
    nextValueIDs.pop_back();
  }

  /// The set of values defined in this scope.
  std::vector<Value> values;

  /// The ID for the next defined value for each region current being
  /// processed in this scope.
  SmallVector<unsigned, 4> nextValueIDs;
};

struct ParsingState {
  ParsingState(Location fileLoc,
               FallbackAsmResourceMap *fallbackResourceMap = nullptr)
      : fallbackResourceMap(fallbackResourceMap), fileLoc(fileLoc),
        wipOperationState(fileLoc, "builtin.unrealized_conversion_cast"),
        // Use the builtin unrealized conversion cast operation to represent
        // forward references to values that aren't yet defined.
        forwardRefOpState(UnknownLoc::get(fileLoc.getContext()),
                          "builtin.unrealized_conversion_cast", ValueRange(),
                          NoneType::get(fileLoc.getContext())) {}

  InFlightDiagnostic emitError(const Twine &msg = {}) {
    return ::emitError(fileLoc, msg);
  }

  Attribute attribute(MlirBytecodeAttrHandle handle) {
    uint64_t i = handle.id;
    if (i >= attributes.size())
      return nullptr;
    if (attributes[i].value)
      return attributes[i].value;
    if (!mlirBytecodeSucceeded(mlirBytecodeParseAttribute(
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

  Type type(MlirBytecodeTypeHandle handle) {
    uint64_t i = handle.id;
    if (i >= types.size())
      return nullptr;
    if (types[i].value)
      return types[i].value;
    if (!mlirBytecodeSucceeded(
            mlirBytecodeParseType(this, (MlirBytecodeTypeHandle){.id = i})))
      return nullptr;
    return types[i].value;
  }

  // These are all public to enable access via plain C functions.
  std::vector<AsmDialectResourceHandle> resources;
  std::vector<BytecodeAttribute> attributes;
  std::vector<BytecodeDialect> dialects;
  std::vector<BytecodeOperationName> opNames;
  std::vector<BytecodeType> types;
  std::vector<StringRef> strings;

  /// Resource parsing.
  DenseMap<StringRef, std::unique_ptr<AsmResourceParser>> resourceParsers;
  FallbackAsmResourceMap *fallbackResourceMap;

  /// Location to use for reporting errors.
  Location fileLoc;

  std::vector<RegionReadState> regionStack;

  /// OperationState used to construct the current operation.
  OperationState wipOperationState;

  /// The current set of available IR value scopes.
  std::vector<ValueScope> valueScopes;
  /// A block containing the set of operations defined to create forward
  /// references.
  Block forwardRefOps;
  /// A block containing previously created, and no longer used, forward
  /// reference operations.
  Block openForwardRefOps;
  /// An operation state used when instantiating forward references.
  OperationState forwardRefOpState;
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

//===----------------------------------------------------------------------===//
// Value Processing

Value createForwardRef(ParsingState &state) {
  // Check for an avaliable existing operation to use. Otherwise, create a new
  // fake operation to use for the reference.
  if (!state.openForwardRefOps.empty()) {
    Operation *op = &state.openForwardRefOps.back();
    op->moveBefore(&state.forwardRefOps, state.forwardRefOps.end());
  } else {
    state.forwardRefOps.push_back(Operation::create(state.forwardRefOpState));
  }
  return state.forwardRefOps.back().getResult(0);
}

LogicalResult defineValues(ParsingState &state, ValueRange newValues) {
  ValueScope &valueScope = state.valueScopes.back();
  std::vector<Value> &values = valueScope.values;

  unsigned &valueID = valueScope.nextValueIDs.back();
  unsigned valueIDEnd = valueID + newValues.size();
  if (valueIDEnd > values.size()) {
    return state.emitError(
               "value index range was outside of the expected range for "
               "the parent region, got [")
           << valueID << ", " << valueIDEnd << "), but the maximum index was "
           << (values.size() - 1);
  }

  // Assign the values and update any forward references.
  for (unsigned i = 0, e = newValues.size(); i != e; ++i, ++valueID) {
    Value newValue = newValues[i];

    // Check to see if a definition for this value already exists.
    if (Value oldValue = std::exchange(values[valueID], newValue)) {
      Operation *forwardRefOp = oldValue.getDefiningOp();

      // Assert that this is a forward reference operation. Given how we compute
      // definition ids (incrementally as we parse), it shouldn't be possible
      // for the value to be defined any other way.
      assert(forwardRefOp && forwardRefOp->getBlock() == &state.forwardRefOps &&
             "value index was already defined?");

      oldValue.replaceAllUsesWith(newValue);
      forwardRefOp->moveBefore(&state.openForwardRefOps,
                               state.openForwardRefOps.end());
    }
  }
  return success();
}

Value parseOperand(ParsingState &state, uint64_t i) {
  std::vector<Value> &values = state.valueScopes.back().values;
  Value &value = values[i];
  // Create a new forward reference if necessary.
  if (!value)
    value = createForwardRef(state);
  return value;
}

} // namespace

/// Wrapper around DialectBytecodeReader invoking C MlirBytecode API.
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
  MlirBytecodeStatus ret = mlirBytecodeDialectReaderReadAPIntWithKnownWidth(
      &reader, bitWidth, malloc, &result);
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

void mlirBytecodeIRSectionEnter(void *callerState, void *retBlock) {
  ParsingState *state = (ParsingState *)callerState;
  Block *ret = (Block *)retBlock;
  ret->dump();
  state->regionStack.emplace_back(MutableArrayRef<Region>{},
                                  /*isIsolatedFromAbove=*/true);
  RegionReadState &readState = state->regionStack.back();
  readState.curBlock = &readState.curBlocks.emplace_back(ret);
  state->valueScopes.emplace_back();
  state->valueScopes.back().push(readState);
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
  state->wipOperationState.location = locAttr;
  state->wipOperationState.name = *opName;
  *opState = &state->wipOperationState;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddAttributeDictionary(
    void *callerState, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeAttrHandle dictHandle) {
  ParsingState *state = (ParsingState *)callerState;
  if (dictHandle.id >= state->attributes.size())
    return mlirBytecodeEmitError(callerState, "out of range attribute handle");
  OperationState *opState = opStateHandle;
  DictionaryAttr attr =
      dyn_cast_if_present<DictionaryAttr>(state->attribute(dictHandle));
  if (!attr)
    return mlirBytecodeEmitError(callerState, "invalid dictionary attribute");
  opState->addAttributes(attr.getValue());
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddResultTypes(
    void *callerState, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeHandlesRef types) {
  ParsingState *state = (ParsingState *)callerState;
  OperationState *opState = opStateHandle;

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

MlirBytecodeStatus mlirBytecodeOperationStateAddOperands(
    void *callerState, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeHandlesRef operands) {
  ParsingState *state = (ParsingState *)callerState;
  OperationState *opState = opStateHandle;

  const uint64_t numOperands = operands.length;
  opState->operands.resize(numOperands);
  for (int i = 0, e = numOperands; i < e; ++i) {
    if (!(opState->operands[i] = parseOperand(*state, i)))
      return mlirBytecodeFailure();
  }
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddRegions(
    void *callerState, MlirBytecodeOperationStateHandle opStateHandle,
    uint64_t numRegions, bool isIsolatedFromAbove) {
  OperationState *opState = opStateHandle;
  opState->regions.reserve(numRegions);
  for (int i = 0, e = numRegions; i < e; ++i)
    opState->regions.push_back(std::make_unique<Region>());

  // TODO: isIsolatedFromAbove

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddSuccessors(
    void *callerState, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeSizesRef successors) {
  ParsingState *state = (ParsingState *)callerState;
  OperationState *opState = opStateHandle;

  const uint64_t numSuccs = successors.length;
  RegionReadState &readState = state->regionStack.back();

  opState->successors.resize(numSuccs);
  for (int i = 0, e = numSuccs; i < e; ++i) {
    if (MLIRBC_UNLIKELY(successors.sizes[i] >= readState.curBlocks.size()))
      return mlirBytecodeEmitError(callerState, "invalid successor index");
    opState->successors[i] = readState.curBlocks[successors.sizes[i]];
  }
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationStatePop(void *callerState,
                              MlirBytecodeOperationStateHandle opStateHandle,
                              MlirBytecodeOperationHandle *opHandle) {
  ParsingState *state = (ParsingState *)callerState;
  OperationState *opState = opStateHandle;

  // Create the operation at the back of the current block.
  Operation *op = Operation::create(*opState);
  *opHandle = op;

  // If the operation had results, update the value references.
  if (op->getNumResults() && failed(defineValues(*state, op->getResults())))
    return mlirBytecodeEmitError(callerState, "invalid operation results");

  // if (!opState->regions.empty()) {
  //   state->regionStack.emplace_back(op, isIsolatedFromAbove);
  //   RegionReadState &readState = state->regionStack.back();
  //   (*readState.curBlock)->push_back(op);
  // }

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationRegionPush(void *callerState,
                                                   MlirBytecodeOperationHandle,
                                                   size_t numBlocks,
                                                   size_t numValues) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeOperationBlockPush(void *callerState,
                                                  MlirBytecodeOperationHandle,
                                                  MlirBytecodeHandlesRef) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeOperationBlockPop(void *callerState,
                                                 MlirBytecodeOperationHandle) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeOperationRegionPop(void *callerState,
                                                  MlirBytecodeOperationHandle) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeParseAttribute(void *callerState,
                                              MlirBytecodeAttrHandle handle) {
  mlirBytecodeEmitDebug("processing attribute %d", (int)handle.id);

  ParsingState *state = (ParsingState *)callerState;
  if (handle.id >= state->attributes.size())
    return mlirBytecodeEmitError(callerState,
                                 "invalid attribute id %" PRIu64 " / %" PRIu64,
                                 handle.id, state->attributes.size());
  BytecodeAttribute &attr = state->attributes[handle.id];
  if (attr.value)
    return mlirBytecodeSuccess();

  if (attr.range.dialectHandle.id >= state->dialects.size())
    return mlirBytecodeEmitError(
        callerState, "invalid dialect id %" PRIu64 " / %" PRIu64,
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
    attr.value = dialect.interface->readAttribute(dialectReader);
    if (!attr.value)
      return mlirBytecodeFailure();
    return mlirBytecodeSuccess();
  }
  auto asmStr =
      StringRef((const char *)attr.range.bytes.data, attr.range.bytes.length);
  // Invoke the MLIR assembly parser to parse the entry text.
  size_t numRead = 0;
  MLIRContext *context = state->fileLoc->getContext();
  attr.value = ::parseAttribute(asmStr, context, numRead);
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

MlirBytecodeStatus mlirBytecodeParseType(void *callerState,
                                         MlirBytecodeTypeHandle handle) {
  ParsingState *state = (ParsingState *)callerState;
  if (handle.id >= state->types.size())
    return mlirBytecodeEmitError(callerState,
                                 "invalid type id %" PRIu64 " / %" PRIu64,
                                 handle.id, state->types.size());
  BytecodeType &type = state->types[handle.id];
  if (type.value)
    return mlirBytecodeSuccess();

  if (type.range.dialectHandle.id >= state->dialects.size())
    return mlirBytecodeEmitError(
        callerState, "invalid dialect id %" PRIu64 " / %" PRIu64,
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

  auto asmStr =
      StringRef((const char *)type.range.bytes.data, type.range.bytes.length);
  // Invoke the MLIR assembly parser to parse the entry text.
  size_t numRead = 0;
  MLIRContext *context = state->fileLoc->getContext();
  type.value = ::parseType(asmStr, context, numRead);
  // Ensure there weren't dangling characters after the entry.
  if (numRead != asmStr.size()) {
    return mlirBytecodeEmitError(
        callerState, "trailing characters found after Type assembly format: %s",
        asmStr.drop_front(numRead).str().c_str());
  }
  if (!type.value)
    return mlirBytecodeFailure();
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeAssociateAttributeRange(
    void *callerState, MlirBytecodeAttrHandle attrHandle,
    MlirBytecodeDialectHandle dialectHandle, MlirBytecodeBytesRef bytes,
    bool hasCustom) {
  ParsingState *state = (ParsingState *)callerState;
  state->attributes[attrHandle.id].range = {
      .bytes = bytes, .hasCustom = hasCustom, .dialectHandle = dialectHandle};
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeAssociateTypeRange(void *callerState,
                               MlirBytecodeTypeHandle typeHandle,
                               MlirBytecodeDialectHandle dialectHandle,
                               MlirBytecodeBytesRef bytes, bool hasCustom) {
  ParsingState *state = (ParsingState *)callerState;
  state->types[typeHandle.id].range = {
      .bytes = bytes, .hasCustom = hasCustom, .dialectHandle = dialectHandle};
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
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeResourceGroupEnter(void *callerState,
                               MlirBytecodeStringHandle groupKey,
                               MlirBytecodeSize numResources) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeResourceBlobCallBack(
    void *callerState, MlirBytecodeStringHandle resourceKey,
    MlirBytecodeStringHandle groupKey, MlirBytecodeBytesRef blob) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeResourceBoolCallBack(
    void *callerState, MlirBytecodeStringHandle resourceKey,
    MlirBytecodeStringHandle groupKey, const uint8_t) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeResourceStringCallBack(
    void *callerState, MlirBytecodeStringHandle resourceKey,
    MlirBytecodeStringHandle groupKey, MlirBytecodeStringHandle) {
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

MlirBytecodeStatus
mlirBytecodeAssociateStringRange(void *callerState,
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
    fprintf(stderr, "MlirBytecodeFailed to open file '%s': %s", argv[1],
            error.message().c_str());
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
  mlir::Block block;
  if (!mlirBytecodeParserStateEmpty(&parserState)) {
    if (mlirBytecodeFailed(mlirBytecodeParse(&state, &parserState, &block)))
      return mlirBytecodeEmitError(&state, "MlirBytecodeFailed to parse file"),
             1;
  }

  return 0;
}
