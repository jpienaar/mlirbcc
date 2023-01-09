//===-- BuiltinParse.c.inc - Builtin attr & type parsers ----------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Builtin Attribute & Type driver.
//===----------------------------------------------------------------------===//

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Entry point for Builtin dialect Attribute parsing.
MlirBytecodeStatus
mlirBytecodeParseBuiltinAttr(void *callerState,
                             MlirBytecodeAttrHandle attrHandle,
                             MlirBytecodeBytesRef str, bool hasCustom);

// Entry point for Builtin dialect Type parsing.
MlirBytecodeStatus
mlirBytecodeParseBuiltinType(void *callerState,
                             MlirBytecodeTypeHandle typeHandle,
                             MlirBytecodeBytesRef str, bool hasCustom);

// NOTE: this is proof-of-concept and just enough working to show end-to-end.
// The parsing of attributes/types (especially those with arrays) will change.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

//===----------------------------------------------------------------------===//
// Create attributes.

// Functions defined by folks adding support of builtin dialect parsing. The
// `attrHandle` is the handle corresponding to the attribute being created. The
// defined function will/can set attribute[attrHandle] if successfully decoded.
static MlirBytecodeStatus mlirBytecodeCreateBuiltinIntegerAttr(
    void *callerState, MlirBytecodeAttrHandle attrHandle,
    MlirBytecodeTypeHandle typeHandle, uint64_t value);
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinStrAttr(void *callerState,
                                 MlirBytecodeAttrHandle attrHandle,
                                 MlirBytecodeStringHandle strHdl);
static MlirBytecodeStatus mlirBytecodeCreateBuiltinFileLineColLoc(
    void *callerState, MlirBytecodeAttrHandle attrHandle,
    MlirBytecodeAttrHandle filename, uint64_t line, uint64_t col);
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinUnknownLoc(void *callerState,
                                    MlirBytecodeAttrHandle attrHandle);

// Handle iterator.
struct MlirBytecodeHandleIterator {
  // Stream over block args.
  MlirBytecodeStream stream;

  // Number of handles in stream.
  const MlirBytecodeSize count;
};
typedef struct MlirBytecodeHandleIterator MlirBytecodeHandleIterator;
typedef struct MlirBytecodeHandleIterator MlirDictionaryHandleIterator;

// Populate the next key and value and increment the iterator.
// Returns whether there was an element.
MlirBytecodeStatus mlirBytecodeGetNextDictionaryHandles(
    void *callerState, MlirDictionaryHandleIterator *iterator,
    MlirBytecodeAttrHandle *name, MlirBytecodeAttrHandle *value);
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinDictionaryAttr(void *callerState,
                                        MlirBytecodeAttrHandle attrHandle,
                                        MlirDictionaryHandleIterator *range);

// Create method for TypeAttr.
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinTypeAttr(void *callerState,
                                  MlirBytecodeAttrHandle bcAttrHandle,
                                  MlirBytecodeTypeHandle value);

//===----------------------------------------------------------------------===//
// Create types.

static MlirBytecodeStatus
mlirBytecodeCreateBuiltinFloat32Type(void *callerState,
                                     MlirBytecodeTypeHandle typeHandle);
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinIndexType(void *callerState,
                                   MlirBytecodeTypeHandle typeHandle);

enum MlirBuiltinSignednessSemantics {
  kBuiltinIntegerTypeSignless, /// No signedness semantics
  kBuiltinIntegerTypeSigned,   /// Signed integer
  kBuiltinIntegerTypeUnsigned, /// Unsigned integer
};
typedef enum MlirBuiltinSignednessSemantics MlirBuiltinSignednessSemantics;

static MlirBytecodeStatus mlirBytecodeCreateBuiltinIntegerType(
    void *callerState, MlirBytecodeTypeHandle typeHandle,
    MlirBuiltinSignednessSemantics signedness, int width);

// TODO
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinFunctionType(void *callerState,
                                      MlirBytecodeTypeHandle typeHandle);
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinRankedTensorType(void *callerState,
                                          MlirBytecodeTypeHandle typeHandle);

//===----------------------------------------------------------------------===//
// Query types.

// Queries the bitwidth of the builtin integer type.
static MlirBytecodeStatus mlirBytecodeQueryBuiltinIntegerTypeWidth(
    void *callerState, MlirBytecodeTypeHandle typeHandle, unsigned *width);

// Create method for UnknownLocAttr.
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinUnknownLocAttr(void *callerState,
                                        MlirBytecodeAttrHandle bcAttrHandle);

// Create method for FlatSymbolRefAttr.
static MlirBytecodeStatus mlirBytecodeCreateBuiltinFlatSymbolRefAttr(
    void *callerState, MlirBytecodeAttrHandle bcAttrHandle,
    MlirBytecodeAttrHandle rootReference);

// Create method for NameLocAttr.
static MlirBytecodeStatus mlirBytecodeCreateBuiltinNameLocAttr(
    void *callerState, MlirBytecodeAttrHandle bcAttrHandle,
    MlirBytecodeAttrHandle name, MlirBytecodeAttrHandle childLoc);

// Create method for StringAttr.
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinStringAttr(void *callerState,
                                    MlirBytecodeAttrHandle bcAttrHandle,
                                    MlirBytecodeStringHandle value);

// Create method for StringAttrWithType.
static MlirBytecodeStatus mlirBytecodeCreateBuiltinStringAttrWithType(
    void *callerState, MlirBytecodeAttrHandle bcAttrHandle,
    MlirBytecodeStringHandle value, MlirBytecodeTypeHandle type);

// Create method for TypeAttr.
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinTypeAttr(void *callerState,
                                  MlirBytecodeAttrHandle bcAttrHandle,
                                  MlirBytecodeTypeHandle value);

// Create method for UnitAttr.
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinUnitAttr(void *callerState,
                                  MlirBytecodeAttrHandle bcAttrHandle);

// Create method for FileLineColLocAttr.
static MlirBytecodeStatus mlirBytecodeCreateBuiltinFileLineColLocAttr(
    void *callerState, MlirBytecodeAttrHandle bcAttrHandle,
    MlirBytecodeAttrHandle file, uint64_t line, uint64_t col);

#ifdef __cplusplus
}
#endif

enum AttributeCode {
  ///   ArrayAttr {
  ///     elements: Attribute[]
  ///   }
  ///
  kBuiltinAttrArrayAttrCode = 0,

  ///   DictionaryAttr {
  ///     attrs: <StringAttr, Attribute>[]
  ///   }
  kBuiltinAttrDictionaryAttrCode = 1,

  ///   StringAttr {
  ///     value: string
  ///   }
  kBuiltinAttrStringAttrCode = 2,

  ///   StringAttrWithType {
  ///     value: string,
  ///     type: Type
  ///   }
  /// A variant of StringAttr with a type.
  kBuiltinAttrStringAttrWithTypeCode = 3,

  ///   FlatSymbolRefAttr {
  ///     rootReference: StringAttr
  ///   }
  /// A variant of SymbolRefAttr with no leaf references.
  kBuiltinAttrFlatSymbolRefAttrCode = 4,

  ///   SymbolRefAttr {
  ///     rootReference: StringAttr,
  ///     leafReferences: FlatSymbolRefAttr[]
  ///   }
  kBuiltinAttrSymbolRefAttrCode = 5,

  ///   TypeAttr {
  ///     value: Type
  ///   }
  kBuiltinAttrTypeAttrCode = 6,

  ///   UnitAttr {
  ///   }
  kBuiltinAttrUnitAttrCode = 7,

  ///   IntegerAttr {
  ///     type: Type
  ///     value: APInt,
  ///   }
  kBuiltinAttrIntegerAttrCode = 8,

  ///   FloatAttr {
  ///     type: FloatType
  ///     value: APFloat
  ///   }
  kBuiltinAttrFloatAttrCode = 9,

  ///   CallSiteLoc {
  ///    callee: LocationAttr,
  ///    caller: LocationAttr
  ///   }
  kBuiltinAttrCallSiteLocCode = 10,

  ///   FileLineColLoc {
  ///     file: StringAttr,
  ///     line: varint,
  ///     column: varint
  ///   }
  kBuiltinAttrFileLineColLocCode = 11,

  ///   FusedLoc {
  ///     locations: LocationAttr[]
  ///   }
  kBuiltinAttrFusedLocCode = 12,

  ///   FusedLocWithMetadata {
  ///     locations: LocationAttr[],
  ///     metadata: Attribute
  ///   }
  /// A variant of FusedLoc with metadata.
  kBuiltinAttrFusedLocWithMetadataCode = 13,

  ///   NameLoc {
  ///     name: StringAttr,
  ///     childLoc: LocationAttr
  ///   }
  kBuiltinAttrNameLocCode = 14,

  ///   UnknownLoc {
  ///   }
  kBuiltinAttrUnknownLocCode = 15,

  ///   DenseResourceElementsAttr {
  ///     type: Type,
  ///     handle: ResourceHandle
  ///   }
  kBuiltinAttrDenseResourceElementsAttrCode = 16,

  ///   DenseArrayAttr {
  ///     type: RankedTensorType,
  ///     data: blob
  ///   }
  kBuiltinAttrDenseArrayAttrCode = 17,

  ///   DenseIntOrFPElementsAttr {
  ///     type: ShapedType,
  ///     data: blob
  ///   }
  kBuiltinAttrDenseIntOrFPElementsAttrCode = 18,

  ///   DenseStringElementsAttr {
  ///     type: ShapedType,
  ///     isSplat: varint,
  ///     data: string[]
  ///   }
  kBuiltinAttrDenseStringElementsAttrCode = 19,

  ///   SparseElementsAttr {
  ///     type: ShapedType,
  ///     indices: DenseIntElementsAttr,
  ///     values: DenseElementsAttr
  ///   }
  kBuiltinAttrSparseElementsAttrCode = 20,
};

/// This enum contains marker codes used to indicate which type is currently
/// being decoded, and how it should be decoded. The order of these codes should
/// generally be unchanged, as any changes will inevitably break compatibility
/// with older bytecode.
enum TypeCode {
  ///   IntegerType {
  ///     widthAndSignedness: varint // (width << 2) | (signedness)
  ///   }
  ///
  kBuiltinTypeIntegerTypeCode = 0,

  ///   IndexType {
  ///   }
  ///
  kBuiltinTypeIndexTypeCode = 1,

  ///   FunctionType {
  ///     inputs: Type[],
  ///     results: Type[]
  ///   }
  ///
  kBuiltinTypeFunctionTypeCode = 2,

  ///   BFloat16Type {
  ///   }
  ///
  kBuiltinTypeBFloat16TypeCode = 3,

  ///   Float16Type {
  ///   }
  ///
  kBuiltinTypeFloat16TypeCode = 4,

  ///   Float32Type {
  ///   }
  ///
  kBuiltinTypeFloat32TypeCode = 5,

  ///   Float64Type {
  ///   }
  ///
  kBuiltinTypeFloat64TypeCode = 6,

  ///   Float80Type {
  ///   }
  ///
  kBuiltinTypeFloat80TypeCode = 7,

  ///   Float128Type {
  ///   }
  ///
  kBuiltinTypeFloat128TypeCode = 8,

  ///   ComplexType {
  ///     elementType: Type
  ///   }
  ///
  kBuiltinTypeComplexTypeCode = 9,

  ///   MemRefType {
  ///     shape: svarint[],
  ///     elementType: Type,
  ///     layout: Attribute
  ///   }
  ///
  kBuiltinTypeMemRefTypeCode = 10,

  ///   MemRefTypeWithMemSpace {
  ///     memorySpace: Attribute,
  ///     shape: svarint[],
  ///     elementType: Type,
  ///     layout: Attribute
  ///   }
  /// Variant of MemRefType with non-default memory space.
  kBuiltinTypeMemRefTypeWithMemSpaceCode = 11,

  ///   NoneType {
  ///   }
  ///
  kBuiltinTypeNoneTypeCode = 12,

  ///   RankedTensorType {
  ///     shape: svarint[],
  ///     elementType: Type,
  ///   }
  ///
  kBuiltinTypeRankedTensorTypeCode = 13,

  ///   RankedTensorTypeWithEncoding {
  ///     encoding: Attribute,
  ///     shape: svarint[],
  ///     elementType: Type
  ///   }
  /// Variant of RankedTensorType with an encoding.
  kBuiltinTypeRankedTensorTypeWithEncodingCode = 14,

  ///   TupleType {
  ///     elementTypes: Type[]
  ///   }
  kBuiltinTypeTupleTypeCode = 15,

  ///   UnrankedMemRefType {
  ///     shape: svarint[]
  ///   }
  ///
  kBuiltinTypeUnrankedMemRefTypeCode = 16,

  ///   UnrankedMemRefTypeWithMemSpace {
  ///     memorySpace: Attribute,
  ///     shape: svarint[]
  ///   }
  /// Variant of UnrankedMemRefType with non-default memory space.
  kBuiltinTypeUnrankedMemRefTypeWithMemSpaceCode = 17,

  ///   UnrankedTensorType {
  ///     elementType: Type
  ///   }
  ///
  kBuiltinTypeUnrankedTensorTypeCode = 18,

  ///   VectorType {
  ///     shape: svarint[],
  ///     elementType: Type
  ///   }
  ///
  kBuiltinTypeVectorTypeCode = 19,

  ///   VectorTypeWithScalableDims {
  ///     numScalableDims: varint,
  ///     shape: svarint[],
  ///     elementType: Type
  ///   }
  /// Variant of VectorType with scalable dimensions.
  kBuiltinTypeVectorTypeWithScalableDimsCode = 20,
};
static MlirBytecodeStatus parseIntegerAttr(void *callerState,
                                           MlirBytecodeAttrHandle attrHandle,
                                           MlirBytecodeStream *pp) {
  MlirBytecodeTypeHandle typeHandle;
  if (mlirBytecodeFailed(mlirBytecodeParseHandle(callerState, pp, &typeHandle)))
    return mlirBytecodeFailure();
  // Example of tricky case: without being able to construct the type, there
  // is nothing to query the bitwidth of the type except to ask the
  // implementation what width the type handle has.
  unsigned width;
  MlirBytecodeStatus ret =
      mlirBytecodeProcessAttribute(callerState, typeHandle);
  if (!mlirBytecodeSucceeded(ret))
    return ret;
  ret =
      mlirBytecodeQueryBuiltinIntegerTypeWidth(callerState, typeHandle, &width);
  if (!mlirBytecodeHandled(ret))
    return ret;

  if (mlirBytecodeFailed(ret))
    return mlirBytecodeEmitError(
        callerState, "expected integer or index type for IntegerAttr");

  // TODO: Handle arbitrary integer sizes.
  if (width > 64)
    return mlirBytecodeUnhandled();

  // FIXME: Actually read APInt.
  int64_t val;
  mlirBytecodeParseSignedVarInt(callerState, pp, &val);

  return mlirBytecodeCreateBuiltinIntegerAttr(callerState, attrHandle,
                                              typeHandle, val);
}

MlirBytecodeStatus mlirBytecodeGetNextDictionaryHandles(
    void *callerState, MlirDictionaryHandleIterator *iterator,
    MlirBytecodeAttrHandle *name, MlirBytecodeAttrHandle *value) {
  if (mlirBytecodeFailed(
          mlirBytecodeParseHandle(callerState, &iterator->stream, name)))
    return mlirBytecodeFailure();
  if (mlirBytecodeFailed(
          mlirBytecodeParseHandle(callerState, &iterator->stream, value)))
    return mlirBytecodeEmitError(callerState, "can't parse value");
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus parseDictionaryAttr(void *callerState,
                                              MlirBytecodeAttrHandle attrHandle,
                                              MlirBytecodeStream *pp) {
  uint64_t size;
  if (mlirBytecodeFailed(mlirBytecodeParseVarInt(callerState, pp, &size)))
    return mlirBytecodeFailure();
  MlirDictionaryHandleIterator iterator = {.stream = *pp, .count = size};
  return mlirBytecodeCreateBuiltinDictionaryAttr(callerState, attrHandle,
                                                 &iterator);
}

static MlirBytecodeStatus parseFileLineColLoc(void *callerState,
                                              MlirBytecodeAttrHandle attrHandle,
                                              MlirBytecodeStream *pp) {
  uint64_t filename, line, col;
  if (mlirBytecodeFailed(mlirBytecodeParseVarInt(callerState, pp, &filename)) ||
      mlirBytecodeFailed(mlirBytecodeParseVarInt(callerState, pp, &line)) ||
      mlirBytecodeFailed(mlirBytecodeParseVarInt(callerState, pp, &col)))
    return mlirBytecodeFailure();
  return mlirBytecodeCreateBuiltinFileLineColLoc(
      callerState, attrHandle, (MlirBytecodeAttrHandle){.id = filename}, line,
      col);
}

static MlirBytecodeStatus parseStringAttr(void *callerState,
                                          MlirBytecodeAttrHandle attrHandle,
                                          MlirBytecodeStream *pp) {
  MlirBytecodeStringHandle strHandle;
  if (mlirBytecodeFailed(mlirBytecodeParseHandle(callerState, pp, &strHandle)))
    return mlirBytecodeFailure();
  return mlirBytecodeCreateBuiltinStrAttr(callerState, attrHandle, strHandle);
}

static MlirBytecodeStatus
parseStringAttrWithType(void *callerState, MlirBytecodeAttrHandle bcAttrHandle,
                        MlirBytecodeStream *bcStream) {
  MlirBytecodeStringHandle value;
  MlirBytecodeTypeHandle type;
  if (mlirBytecodeSucceeded(
          mlirBytecodeParseHandle(callerState, bcStream, &value)) &&
      mlirBytecodeSucceeded(
          mlirBytecodeParseHandle(callerState, bcStream, &type))) {
    return mlirBytecodeCreateBuiltinStringAttrWithType(
        callerState, bcAttrHandle, value, type);
  }
  return mlirBytecodeEmitError(callerState, "invalid StringAttrWithType");
}

static MlirBytecodeStatus parseTypeAttr(void *callerState,
                                        MlirBytecodeAttrHandle bcAttrHandle,
                                        MlirBytecodeStream *bcStream) {
  MlirBytecodeTypeHandle value;
  if (mlirBytecodeSucceeded(
          mlirBytecodeParseHandle(callerState, bcStream, &value)))
    return mlirBytecodeCreateBuiltinTypeAttr(callerState, bcAttrHandle, value);

  return mlirBytecodeEmitError(callerState, "invalid TypeAttr");
}

MlirBytecodeStream mlirBytecodeStreamCreate(MlirBytecodeBytesRef bytes) {
  MlirBytecodeStream ret;
  ret.start = ret.pos = bytes.data;
  ret.end = ret.start + bytes.length;
  return ret;
}

MlirBytecodeStatus
mlirBytecodeParseBuiltinAttr(void *callerState,
                             MlirBytecodeAttrHandle attrHandle,
                             MlirBytecodeBytesRef str, bool hasCustom) {
  if (!hasCustom)
    return mlirBytecodeUnhandled();

  MlirBytecodeStream stream = mlirBytecodeStreamCreate(str);
  MlirBytecodeStream *pp = &stream;

  uint64_t kind;
  if (mlirBytecodeFailed(mlirBytecodeParseVarInt(callerState, pp, &kind)))
    return mlirBytecodeFailure();
  mlirBytecodeEmitDebug("builtin attr kind %d", (int)kind);

  switch (kind) {
  case kBuiltinAttrIntegerAttrCode:
    return parseIntegerAttr(callerState, attrHandle, pp);
  case kBuiltinAttrTypeAttrCode:
    return parseTypeAttr(callerState, attrHandle, pp);
  case kBuiltinAttrDictionaryAttrCode:
    return parseDictionaryAttr(callerState, attrHandle, pp);
  case kBuiltinAttrStringAttrCode:
    return parseStringAttr(callerState, attrHandle, pp);
  case kBuiltinAttrStringAttrWithTypeCode:
    return parseStringAttrWithType(callerState, attrHandle, pp);
  case kBuiltinAttrUnknownLocCode:
    return mlirBytecodeCreateBuiltinUnknownLoc(callerState, attrHandle);
  case kBuiltinAttrFileLineColLocCode:
    return parseFileLineColLoc(callerState, attrHandle, pp);
  default:
    mlirBytecodeEmitDebug("missing parsing for builtin attr %" PRIu64 "\n",
                          kind);
    return mlirBytecodeUnhandled();
  }
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus parseFloat32Type(void *callerState,
                                           MlirBytecodeAttrHandle typeHandle,
                                           MlirBytecodeStream *pp) {
  return mlirBytecodeCreateBuiltinFloat32Type(callerState, typeHandle);
}

static MlirBytecodeStatus parseIndexType(void *callerState,
                                         MlirBytecodeAttrHandle typeHandle,
                                         MlirBytecodeStream *pp) {
  return mlirBytecodeCreateBuiltinIndexType(callerState, typeHandle);
}

static MlirBytecodeStatus parseIntegerType(void *callerState,
                                           MlirBytecodeTypeHandle bcTypeHandle,
                                           MlirBytecodeStream *bcStream) {
  uint64_t _widthAndSignedness, signedness, width;
  if (mlirBytecodeSucceeded(mlirBytecodeParseVarInt(callerState, bcStream,
                                                    &_widthAndSignedness)) &&
      mlirBytecodeSucceeded(
          ((signedness = _widthAndSignedness & 0x3), mlirBytecodeSuccess())) &&
      mlirBytecodeSucceeded(
          ((width = _widthAndSignedness >> 2), mlirBytecodeSuccess()))) {
    return mlirBytecodeCreateBuiltinIntegerType(
        callerState, bcTypeHandle, (MlirBuiltinSignednessSemantics)signedness,
        width);
  }
  return mlirBytecodeEmitError(callerState, "invalid IntegerType");
}

static MlirBytecodeStatus parseFunctionType(void *callerState,
                                            MlirBytecodeAttrHandle typeHandle,
                                            MlirBytecodeStream *pp) {
  // FIXME
  return mlirBytecodeCreateBuiltinFunctionType(callerState, typeHandle);
}

static MlirBytecodeStatus
parseRankedTensorType(void *callerState, MlirBytecodeAttrHandle typeHandle,
                      MlirBytecodeStream *pp) {
  // FIXME
  return mlirBytecodeCreateBuiltinRankedTensorType(callerState, typeHandle);
}

MlirBytecodeStatus
mlirBytecodeParseBuiltinType(void *callerState,
                             MlirBytecodeTypeHandle typeHandle,
                             MlirBytecodeBytesRef str, bool hasCustom) {
  mlirBytecodeEmitDebug("type str = %s %d", str.data, hasCustom);
  if (!hasCustom)
    return mlirBytecodeUnhandled();

  MlirBytecodeStream stream = mlirBytecodeStreamCreate(str);
  uint64_t kind;
  if (mlirBytecodeFailed(mlirBytecodeParseVarInt(callerState, &stream, &kind)))
    return mlirBytecodeFailure();
  mlirBytecodeEmitDebug("builtin type kind %d", (int)kind);

  switch (kind) {
  case kBuiltinTypeIntegerTypeCode:
    return parseIntegerType(callerState, typeHandle, &stream);
  case kBuiltinTypeRankedTensorTypeCode:
    return parseRankedTensorType(callerState, typeHandle, &stream);
  case kBuiltinTypeIndexTypeCode:
    return parseIndexType(callerState, typeHandle, &stream);
  case kBuiltinTypeFloat32TypeCode:
    return parseFloat32Type(callerState, typeHandle, &stream);
  case kBuiltinTypeFunctionTypeCode:
    return parseFunctionType(callerState, typeHandle, &stream);
    break;
  default:
    mlirBytecodeEmitDebug("missing parsing for builtin type %" PRIu64 "\n",
                          kind);
    return mlirBytecodeUnhandled();
  }

  return mlirBytecodeSuccess();
}
#pragma GCC diagnostic pop
