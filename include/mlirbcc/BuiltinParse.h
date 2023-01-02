//===-- BuiltinParse.h - C parser driver for Builtin dialect ------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Builtin dialect Attribute & Type driver.
//===----------------------------------------------------------------------===//

// The implementation of the dialect parsing is expanded if
//   #define MLIRBC_PARSE_IMPLEMENTATION
// or
//   #define MLIRBC_BUILTIN_PARSE_IMPLEMENTATION
// is set before including this file.

#ifndef MLIRBC_BUILTIN_PARSE
#define MLIRBC_BUILTIN_PARSE
#ifdef __cplusplus
extern "C" {
#endif


// Entry point for Builtin dialect Attribute parsing.
MlirBytecodeStatus
mlirBytecodeParseBuiltinAttr(void *state, MlirBytecodeDialectHandle dialectHdl,
                             MlirBytecodeAttrHandle attrHdl, size_t total,
                             bool hasCustom, MlirBytecodeBytesRef str);

// Entry point for Builtin dialect Type parsing.
MlirBytecodeStatus
mlirBytecodeParseBuiltinType(void *state, MlirBytecodeDialectHandle dialectHdl,
                             MlirBytecodeTypeHandle typeHdl, size_t total,
                             bool hasCustom, MlirBytecodeBytesRef str);

enum MlirBuiltinSignednessSemantics {
  kBuiltinIntegerTypeSignless, /// No signedness semantics
  kBuiltinIntegerTypeSigned,   /// Signed integer
  kBuiltinIntegerTypeUnsigned, /// Unsigned integer
};
typedef enum MlirBuiltinSignednessSemantics MlirBuiltinSignednessSemantics;

//===----------------------------------------------------------------------===//
// Create attributes.

// Functions defined by folks adding support of builtin dialect parsing. The
// `attrHdl` is the handle corresponding to the attribute being created. The
// defined function will/can set attribute[attrHdl] if successfully decoded.
extern MlirBytecodeStatus mlirBytecodeCreateBuiltinIntegerAttr(
    void *state, MlirBytecodeAttrHandle attrHdl, MlirBytecodeTypeHandle typeHdl,
    uint64_t value);
extern MlirBytecodeStatus
mlirBytecodeCreateBuiltinStrAttr(void *state, MlirBytecodeAttrHandle attrHdl,
                                 MlirBytecodeStringHandle strHdl);
extern MlirBytecodeStatus mlirBytecodeCreateBuiltinFileLineColLoc(
    void *state, MlirBytecodeAttrHandle attrHdl,
    MlirBytecodeAttrHandle filename, uint64_t line, uint64_t col);
extern MlirBytecodeStatus
mlirBytecodeCreateBuiltinUnknownLoc(void *state,
                                    MlirBytecodeAttrHandle attrHdl);

typedef struct MlirBytecodeHandleIterator MlirDictionaryHandleIterator;
// Populate the next key and value and increment the iterator.
// Returns whether there was an element.
MlirBytecodeStatus
mlirBytecodeGetNextDictionaryHandles(MlirDictionaryHandleIterator *iterator,
                                     MlirBytecodeAttrHandle *name,
                                     MlirBytecodeAttrHandle *value);
extern MlirBytecodeStatus
mlirBytecodeCreateBuiltinDictionaryAttr(void *state,
                                        MlirBytecodeAttrHandle attrHdl,
                                        MlirDictionaryHandleIterator *range);

// Create method for TypeAttr.
extern MlirBytecodeStatus
mlirBytecodeCreateBuiltinTypeAttr(void *bcUserState,
                                  MlirBytecodeAttrHandle bcAttrHandle,
                                  MlirBytecodeTypeHandle value);

//===----------------------------------------------------------------------===//
// Create types.

extern MlirBytecodeStatus
mlirBytecodeCreateBuiltinFloat32Type(void *state,
                                     MlirBytecodeTypeHandle typeHdl);
extern MlirBytecodeStatus
mlirBytecodeCreateBuiltinIndexType(void *state, MlirBytecodeTypeHandle typeHdl);
extern MlirBytecodeStatus mlirBytecodeCreateBuiltinIntegerType(
    void *state, MlirBytecodeTypeHandle typeHdl,
    MlirBuiltinSignednessSemantics signedness, int width);

// TODO
extern MlirBytecodeStatus
mlirBytecodeCreateBuiltinFunctionType(void *state,
                                      MlirBytecodeTypeHandle typeHdl);
extern MlirBytecodeStatus
mlirBytecodeCreateBuiltinRankedTensorType(void *state,
                                          MlirBytecodeTypeHandle typeHdl);

//===----------------------------------------------------------------------===//
// Query types.

// Queries the bitwidth of the builtin integer type.
extern MlirBytecodeStatus mlirBytecodeQueryBuiltinIntegerTypeWidth(
    void *state, MlirBytecodeTypeHandle typeHandle, unsigned *width);

// Entry point for Builtin dialect Attribute parsing.
MlirBytecodeStatus
mlirBytecodeParseBuiltinAttr(void *state,
                             MlirBytecodeDialectHandle dialectHandle,
                             MlirBytecodeAttrHandle attrHandle, size_t total,
                             bool hasCustom, MlirBytecodeBytesRef str);

// Entry point for Builtin dialect Type parsing.
MlirBytecodeStatus
mlirBytecodeParseBuiltinType(void *state,
                             MlirBytecodeDialectHandle dialectHandle,
                             MlirBytecodeTypeHandle typeHandle, size_t total,
                             bool hasCustom, MlirBytecodeBytesRef str);

// Create method for UnknownLocAttr.
extern MlirBytecodeStatus
mlirBytecodeCreateBuiltinUnknownLocAttr(void *bcUserState,
                                        MlirBytecodeAttrHandle bcAttrHandle);

// Create method for FlatSymbolRefAttr.
extern MlirBytecodeStatus mlirBytecodeCreateBuiltinFlatSymbolRefAttr(
    void *bcUserState, MlirBytecodeAttrHandle bcAttrHandle,
    MlirBytecodeAttrHandle rootReference);

// Create method for NameLocAttr.
extern MlirBytecodeStatus mlirBytecodeCreateBuiltinNameLocAttr(
    void *bcUserState, MlirBytecodeAttrHandle bcAttrHandle,
    MlirBytecodeAttrHandle name, MlirBytecodeAttrHandle childLoc);

// Create method for StringAttr.
extern MlirBytecodeStatus
mlirBytecodeCreateBuiltinStringAttr(void *bcUserState,
                                    MlirBytecodeAttrHandle bcAttrHandle,
                                    MlirBytecodeStringHandle value);

// Create method for StringAttrWithType.
extern MlirBytecodeStatus mlirBytecodeCreateBuiltinStringAttrWithType(
    void *bcUserState, MlirBytecodeAttrHandle bcAttrHandle,
    MlirBytecodeStringHandle value, MlirBytecodeTypeHandle type);

// Create method for TypeAttr.
extern MlirBytecodeStatus
mlirBytecodeCreateBuiltinTypeAttr(void *bcUserState,
                                  MlirBytecodeAttrHandle bcAttrHandle,
                                  MlirBytecodeTypeHandle value);

// Create method for UnitAttr.
extern MlirBytecodeStatus
mlirBytecodeCreateBuiltinUnitAttr(void *bcUserState,
                                  MlirBytecodeAttrHandle bcAttrHandle);

// Create method for FileLineColLocAttr.
extern MlirBytecodeStatus mlirBytecodeCreateBuiltinFileLineColLocAttr(
    void *bcUserState, MlirBytecodeAttrHandle bcAttrHandle,
    MlirBytecodeAttrHandle file, uint64_t line, uint64_t col);

#ifdef __cplusplus
}
#endif
#endif // MLIRBC_BUILTIN_PARSE

#if defined(MLIRBC_PARSE_IMPLEMENTATION) ||                                    \
    defined(MLIRBC_MLIRBC_BUILTIN_PARSE_IMPLEMENTATION)
#include <inttypes.h>

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
static MlirBytecodeStatus parseIntegerAttr(void *state,
                                           MlirBytecodeAttrHandle attrHandle,
                                           MlirBytecodeStream *pp) {
  MlirBytecodeTypeHandle typeHandle;
  if (mlirBytecodeFailed(mlirBytecodeReadHandle(pp, &typeHandle)))
    return mlirBytecodeFailure();
  // Example of tricky case: without being able to construct the type, there
  // is nothing to query the bitwidth of the type except to ask the
  // implementation what width the type handle has.
  unsigned width;
  MlirBytecodeStatus ret =
      mlirBytecodeQueryBuiltinIntegerTypeWidth(state, typeHandle, &width);
  if (!mlirBytecodeHandled(ret))
    return ret;

  if (mlirBytecodeFailed(ret)) {
    return mlirBytecodeEmitError(
        "expected integer or index type for IntegerAttr");
  }

  // TODO: Handle arbitrary integer sizes.
  if (width > 64)
    return mlirBytecodeUnhandled();

  // FIXME: Actually read APInt.
  int64_t val;
  mlirBytecodeReadSignedVarInt(pp, &val);

  return mlirBytecodeCreateBuiltinIntegerAttr(state, attrHandle, typeHandle,
                                              val);
}

MlirBytecodeStatus
mlirBytecodeGetNextDictionaryHandles(MlirDictionaryHandleIterator *iterator,
                                     MlirBytecodeAttrHandle *name,
                                     MlirBytecodeAttrHandle *value) {
  if (mlirBytecodeFailed(mlirBytecodeReadHandle(&iterator->stream, name)))
    return mlirBytecodeFailure();
  if (mlirBytecodeFailed(mlirBytecodeReadHandle(&iterator->stream, value)))
    return mlirBytecodeEmitError("can't parse value");
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus parseDictionaryAttr(void *state,
                                              MlirBytecodeAttrHandle attrHandle,
                                              MlirBytecodeStream *pp) {
  MlirBytecodeSize size;
  if (mlirBytecodeFailed(mlirBytecodeReadVarInt(pp, &size)))
    return mlirBytecodeFailure();
  MlirDictionaryHandleIterator iterator = {.stream = *pp, .count = size};
  return mlirBytecodeCreateBuiltinDictionaryAttr(state, attrHandle, &iterator);
}

static MlirBytecodeStatus parseFileLineColLoc(void *state,
                                              MlirBytecodeAttrHandle attrHandle,
                                              MlirBytecodeStream *pp) {
  uint64_t filename, line, col;
  if (mlirBytecodeFailed(mlirBytecodeReadVarInt(pp, &filename)) ||
      mlirBytecodeFailed(mlirBytecodeReadVarInt(pp, &line)) ||
      mlirBytecodeFailed(mlirBytecodeReadVarInt(pp, &col)))
    return mlirBytecodeFailure();
  return mlirBytecodeCreateBuiltinFileLineColLoc(
      state, attrHandle, (MlirBytecodeAttrHandle){.id = filename}, line, col);
}

static MlirBytecodeStatus parseStringAttr(void *state,
                                          MlirBytecodeAttrHandle attrHandle,
                                          MlirBytecodeStream *pp) {
  MlirBytecodeStringHandle strHandle;
  if (mlirBytecodeFailed(mlirBytecodeReadHandle(pp, &strHandle)))
    return mlirBytecodeFailure();
  return mlirBytecodeCreateBuiltinStrAttr(state, attrHandle, strHandle);
}

static MlirBytecodeStatus
parseStringAttrWithType(void *bcUserState, MlirBytecodeAttrHandle bcAttrHandle,
                        MlirBytecodeStream *bcStream) {
  MlirBytecodeStringHandle value;
  MlirBytecodeTypeHandle type;
  if (mlirBytecodeSucceeded(mlirBytecodeReadHandle(bcStream, &value)) &&
      mlirBytecodeSucceeded(mlirBytecodeReadHandle(bcStream, &type))) {
    return mlirBytecodeCreateBuiltinStringAttrWithType(
        bcUserState, bcAttrHandle, value, type);
  }
  return mlirBytecodeEmitError("invalid StringAttrWithType");
}

static MlirBytecodeStatus parseTypeAttr(void *bcUserState,
                                        MlirBytecodeAttrHandle bcAttrHandle,
                                        MlirBytecodeStream *bcStream) {
  MlirBytecodeTypeHandle value;
  if (mlirBytecodeSucceeded(mlirBytecodeReadHandle(bcStream, &value)))
    return mlirBytecodeCreateBuiltinTypeAttr(bcUserState, bcAttrHandle, value);

  return mlirBytecodeEmitError("invalid TypeAttr");
}

MlirBytecodeStatus
mlirBytecodeParseBuiltinAttr(void *state, MlirBytecodeDialectHandle dialectHdl,
                             MlirBytecodeAttrHandle attrHandle, size_t total,
                             bool hasCustom, MlirBytecodeBytesRef str) {
  if (!hasCustom)
    return mlirBytecodeUnhandled();

  MlirBytecodeStream stream = mlirBytecodeStreamCreate(str);
  MlirBytecodeStream *pp = &stream;

  uint64_t kind;
  if (mlirBytecodeFailed(mlirBytecodeReadVarInt(pp, &kind)))
    return mlirBytecodeFailure();
  mlirBytecodeEmitDebug("builtin attr kind %d", (int)kind);

  switch (kind) {
    // TODO: All should be single return calling a parse function, each parse
    // function should end with call in extern function.
  case kBuiltinAttrIntegerAttrCode:
    return parseIntegerAttr(state, attrHandle, pp);
  case kBuiltinAttrTypeAttrCode:
    return parseTypeAttr(state, attrHandle, pp);
  case kBuiltinAttrDictionaryAttrCode:
    return parseDictionaryAttr(state, attrHandle, pp);
  case kBuiltinAttrStringAttrCode:
    return parseStringAttr(state, attrHandle, pp);
  case kBuiltinAttrStringAttrWithTypeCode:
    return parseStringAttrWithType(state, attrHandle, pp);
  case kBuiltinAttrUnknownLocCode:
    return mlirBytecodeCreateBuiltinUnknownLoc(state, attrHandle);
  case kBuiltinAttrFileLineColLocCode:
    return parseFileLineColLoc(state, attrHandle, pp);
  default:
    // mlirBytecodeEmitError("missing parsing for Builtin attr %" PRIu64 "\n",
    // kind);
    return mlirBytecodeUnhandled();
  }
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus parseFloat32Type(void *state,
                                           MlirBytecodeAttrHandle typeHandle,
                                           MlirBytecodeStream *pp) {
  return mlirBytecodeCreateBuiltinFloat32Type(state, typeHandle);
}

static MlirBytecodeStatus parseIndexType(void *state,
                                         MlirBytecodeAttrHandle typeHandle,
                                         MlirBytecodeStream *pp) {
  return mlirBytecodeCreateBuiltinIndexType(state, typeHandle);
}

static MlirBytecodeStatus parseIntegerType(void *bcUserState,
                                           MlirBytecodeTypeHandle bcTypeHandle,
                                           MlirBytecodeStream *bcStream) {
  uint64_t _widthAndSignedness, signedness, width;
  if (mlirBytecodeSucceeded(
          mlirBytecodeReadVarInt(bcStream, &_widthAndSignedness)) &&
      mlirBytecodeSucceeded(
          ((signedness = _widthAndSignedness & 0x3), mlirBytecodeSuccess())) &&
      mlirBytecodeSucceeded(
          ((width = _widthAndSignedness >> 2), mlirBytecodeSuccess()))) {
    return mlirBytecodeCreateBuiltinIntegerType(bcUserState, bcTypeHandle,
                                                signedness, width);
  }
  return mlirBytecodeEmitError("invalid IntegerType");
}

static MlirBytecodeStatus parseFunctionType(void *state,
                                            MlirBytecodeAttrHandle typeHandle,
                                            MlirBytecodeStream *pp) {
  // FIXME
  return mlirBytecodeCreateBuiltinFunctionType(state, typeHandle);
}

static MlirBytecodeStatus
parseRankedTensorType(void *state, MlirBytecodeAttrHandle typeHandle,
                      MlirBytecodeStream *pp) {
  // FIXME
  return mlirBytecodeCreateBuiltinRankedTensorType(state, typeHandle);
}

MlirBytecodeStatus
mlirBytecodeParseBuiltinType(void *state, MlirBytecodeDialectHandle dialectHdl,
                             MlirBytecodeTypeHandle typeHdl, size_t total,
                             bool hasCustom, MlirBytecodeBytesRef str) {
  if (!hasCustom)
    return mlirBytecodeUnhandled();

  MlirBytecodeStream stream = mlirBytecodeStreamCreate(str);
  uint64_t kind;
  if (mlirBytecodeFailed(mlirBytecodeReadVarInt(&stream, &kind)))
    return mlirBytecodeFailure();

  switch (kind) {
  case kBuiltinTypeIntegerTypeCode:
    return parseIntegerType(state, typeHdl, &stream);
  case kBuiltinTypeRankedTensorTypeCode:
    return parseRankedTensorType(state, typeHdl, &stream);
  case kBuiltinTypeIndexTypeCode:
    return parseIndexType(state, typeHdl, &stream);
  case kBuiltinTypeFloat32TypeCode:
    return parseFloat32Type(state, typeHdl, &stream);
  case kBuiltinTypeFunctionTypeCode:
    return parseFunctionType(state, typeHdl, &stream);
    break;
  default:
    return mlirBytecodeUnhandled();
  }

  return mlirBytecodeSuccess();
}

#endif // MLIRBC_PARSE_IMPLEMENTATION or
       // MLIRBC_MLIRBC_BUILTIN_PARSE_IMPLEMENTATION