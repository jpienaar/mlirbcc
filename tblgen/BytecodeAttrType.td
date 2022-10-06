// Bytecode base classes/defs.

class Bytecode<string t, string p> {
  string cType = t;
  // {0} == user state
  // {1} == bytecode stream
  // {2} == variable being parsed
  string cParser = p;
}

def Attr : Bytecode<"MlirBytecodeAttrHandle", "mlirBytecodeParseAttr({1}, &{2})">;
def Type: Bytecode<"MlirBytecodeTypeHandle", "mlirBytecodeParseType({1}, &{2})">;
def VarInt : Bytecode<"uint64_t", "mlirBytecodeReadVarInt({1}, &{2})">;

class KnownWidthAPInt<string s> : Bytecode<
  "MlirBytecodeAPInt",
  "mlirBytecodeParseAPIntKnownWidth({0}, {1}, " # s # ", &{2})">;

// Helper to define a variable that is defined later but not parsed.
class NoParseSuccess<string t, string d> : Bytecode<t,
  "(({2} = " # d # "), mlirBytecodeSuccess())">;

// Array instances.
def ArrayTypes: Bytecode<"MlirBytecodeArrayHandle", "mlirBytecodeParseTypes({1}, &{2})">;
def ArrayAttrs: Bytecode<"MlirBytecodeArrayHandle", "mlirBytecodeParseAttrs({1}, &{2})">;

// Define dialect attribute or type.
// - Any members starting with underscore is not fed to create function but treated as local variable.
class DialectAttributeOrType<string di, int e, dag d> {
  int enum = e;
  dag members = d;
  string dialect = di;

  // Any post-processing that needs to be done.
  code postProcess = "";
}

class DialectAttribute<string di, int e, dag d> : DialectAttributeOrType<di, e, d>,
  Bytecode<Attr.cType, Attr.cParser>;
class DialectType<string di, int e, dag d> : DialectAttributeOrType<di, e, d>,
  Bytecode<Type.cType, Type.cParser>;

def attr;
def type;

// ----------------
// Builtin dialect.
// ================

class BuiltinDialectAttribute<int e, dag d> : DialectAttribute<"Builtin", e, d>;
class BuiltinDialectType<int e, dag d> : DialectType<"Builtin", e, d>;

defvar LocationAttr = Attr;

def String : Bytecode<"MlirBytecodeStringHandle", "mlirBytecodeParseString({1}, &{2})">;

// enum AttributeCode {
//  ///   ArrayAttr {
//  ///     elements: Attribute[]
//  ///   }
//  ///
//  kArrayAttr = 0,
//
//  ///   DictionaryAttr {
//  ///     attrs: <StringAttr, Attribute>[]
//  ///   }
//  kDictionaryAttr = 1,
//
//  ///   StringAttr {
//  ///     value: string
//  ///   }
//  kStringAttr = 2,
def StringAttr : BuiltinDialectAttribute</*enum=*/2, (attr
  String:$value
)>;

//  ///   StringAttrWithType {
//  ///     value: string,
//  ///     type: Type
//  ///   }
//  /// A variant of StringAttr with a type.
//  kStringAttrWithType = 3,
def StringAttrWithType : BuiltinDialectAttribute</*enum=*/3, (attr
  String:$value,
  Type:$type
)>;

//  ///   FlatSymbolRefAttr {
//  ///     rootReference: StringAttr
//  ///   }
//  /// A variant of SymbolRefAttr with no leaf references.
//  kFlatSymbolRefAttr = 4,
def FlatSymbolRefAttr: BuiltinDialectAttribute</*enum=*/4, (attr
  StringAttr:$rootReference
)>;

//  ///   SymbolRefAttr {
//  ///     rootReference: StringAttr,
//  ///     leafReferences: FlatSymbolRefAttr[]
//  ///   }
//  kSymbolRefAttr = 5,
//
//  ///   TypeAttr {
//  ///     value: Type
//  ///   }
//  kTypeAttr = 6,
def TypeAttr: BuiltinDialectAttribute</*enum=*/6, (attr
  Type:$value
)>;

//  ///   UnitAttr {
//  ///   }
//  kUnitAttr = 7,
def UnitAttr: BuiltinDialectAttribute</*enum=*/7, (attr)>;

//  ///   IntegerAttr {
//  ///     type: Type
//  ///     value: APInt,
//  ///   }
//  kIntegerAttr = 8,
def IntegerAttr: BuiltinDialectAttribute</*enum=*/8, (attr
  Type:$type,
  Bytecode<"uint64_t", "mlirBytecodeQueryBuiltinIntegerTypeWidth({0}, type, &{2})">:$_width,
  KnownWidthAPInt<"_width">:$value
)>;

//
//  ///   FloatAttr {
//  ///     type: FloatType
//  ///     value: APFloat
//  ///   }
//  kFloatAttr = 9,
//
//  ///   CallSiteLoc {
//  ///    callee: LocationAttr,
//  ///    caller: LocationAttr
//  ///   }
//  kCallSiteLoc = 10,
def CallSiteLocAttr : BuiltinDialectAttribute</*enum=*/10, (attr
  LocationAttr:$callee,
  LocationAttr:$caller
)>;

//  ///   FileLineColLoc {
//  ///     file: StringAttr,
//  ///     line: varint,
//  ///     column: varint
//  ///   }
//  kFileLineColLoc = 11,
def FileLineColLocAttr : BuiltinDialectAttribute</*enum=*/11, (attr
  Attr:$file,
  VarInt:$line,
  VarInt:$col
)>;

//  ///   FusedLoc {
//  ///     locations: LocationAttr[]
//  ///   }
//  kFusedLoc = 12,
def FusedLocAttr : BuiltinDialectAttribute</*enum=*/12, (attr
  ArrayAttrs:$locations
)>;

//  ///   FusedLocWithMetadata {
//  ///     locations: LocationAttr[],
//  ///     metadata: Attribute
//  ///   }
//  /// A variant of FusedLoc with metadata.
//  kFusedLocWithMetadata = 13,
def FusedLocWithMetadataAttr : BuiltinDialectAttribute</*enum=*/13, (attr
  ArrayAttrs:$locations,
  Attr:$metadata
)>;

//  ///   NameLoc {
//  ///     name: StringAttr,
//  ///     childLoc: LocationAttr
//  ///   }
//  kNameLoc = 14,
def NameLocAttr : BuiltinDialectAttribute</*enum=*/14, (attr
  StringAttr:$name,
  LocationAttr:$childLoc
)>;

//  ///   UnknownLoc {
//  ///   }
//  kUnknownLoc = 15,
def UnknownLocAttr : BuiltinDialectAttribute</*enum=*/15, (attr)>;

//  ///   DenseResourceElementsAttr {
//  ///     type: Type,
//  ///     handle: ResourceHandle
//  ///   }
//  kDenseResourceElementsAttr = 16,
//
//  ///   DenseArrayAttr {
//  ///     type: RankedTensorType,
//  ///     data: blob
//  ///   }
//  kDenseArrayAttr = 17,
//
//  ///   DenseIntOrFPElementsAttr {
//  ///     type: ShapedType,
//  ///     data: blob
//  ///   }
//  kDenseIntOrFPElementsAttr = 18,
//
//  ///   DenseStringElementsAttr {
//  ///     type: ShapedType,
//  ///     isSplat: varint,
//  ///     data: string[]
//  ///   }
//  kDenseStringElementsAttr = 19,
//
//  ///   SparseElementsAttr {
//  ///     type: ShapedType,
//  ///     indices: DenseIntElementsAttr,
//  ///     values: DenseElementsAttr
//  ///   }
//  kSparseElementsAttr = 20,

// Types
// -----

// enum TypeCode {
//   ///   IntegerType {
//   ///     widthAndSignedness: varint // (width << 2) | (signedness)
//   ///   }
//   ///
//   kIntegerType = 0,
def IntegerType : BuiltinDialectType</*enum=*/0, (type
  VarInt:$_widthAndSignedness,
  // Split up parsed varint.
  NoParseSuccess<"uint64_t", "_widthAndSignedness & 0x3">:$signedness,
  NoParseSuccess<"uint64_t", "_widthAndSignedness >> 2">:$width
)>;

// 
//   ///   IndexType {
//   ///   }
//   ///
//   kIndexType = 1,
def IndexType : BuiltinDialectType</*enum=*/1, (type)>;

//   ///   FunctionType {
//   ///     inputs: Type[],
//   ///     results: Type[]
//   ///   }
//   ///
//   kFunctionType = 2,
def FunctionType : BuiltinDialectType</*enum=*/2, (type
  ArrayTypes:$inputs,
  ArrayTypes:$results
)>;

//   ///   BFloat16Type {
//   ///   }
//   ///
//   kBFloat16Type = 3,
def Bfloat16Type : BuiltinDialectType</*enum=*/3, (type)>;

//   ///   Float16Type {
//   ///   }
//   ///
//   kFloat16Type = 4,
def Float16Type : BuiltinDialectType</*enum=*/4, (type)>;

//   ///   Float32Type {
//   ///   }
//   ///
//   kFloat32Type = 5,
def Float32Type : BuiltinDialectType</*enum=*/5, (type)>;

//   ///   Float64Type {
//   ///   }
//   ///
//   kFloat64Type = 6,
def Float64Type : BuiltinDialectType</*enum=*/6, (type)>;

//   ///   Float80Type {
//   ///   }
//   ///
//   kFloat80Type = 7,
def Float80Type : BuiltinDialectType</*enum=*/7, (type)>;

//   ///   Float128Type {
//   ///   }
//   ///
//   kFloat128Type = 8,
def Float128Type : BuiltinDialectType</*enum=*/8, (type)>;

//   ///   ComplexType {
//   ///     elementType: Type
//   ///   }
//   ///
//   kComplexType = 9,
def ComplexType : BuiltinDialectType</*enum=*/9, (type
  Type:$elementType
)>;

//   ///   MemRefType {
//   ///     shape: svarint[],
//   ///     elementType: Type,
//   ///     layout: Attribute
//   ///   }
//   ///
//   kMemRefType = 10,
// 
//   ///   MemRefTypeWithMemSpace {
//   ///     memorySpace: Attribute,
//   ///     shape: svarint[],
//   ///     elementType: Type,
//   ///     layout: Attribute
//   ///   }
//   /// Variant of MemRefType with non-default memory space.
//   kMemRefTypeWithMemSpace = 11,
// 
//   ///   NoneType {
//   ///   }
//   ///
//   kNoneType = 12,
def NoneType : BuiltinDialectType</*enum=*/12, (type)>;

//   ///   RankedTensorType {
//   ///     shape: svarint[],
//   ///     elementType: Type,
//   ///   }
//   ///
//   kRankedTensorType = 13,
// 
//   ///   RankedTensorTypeWithEncoding {
//   ///     encoding: Attribute,
//   ///     shape: svarint[],
//   ///     elementType: Type
//   ///   }
//   /// Variant of RankedTensorType with an encoding.
//   kRankedTensorTypeWithEncoding = 14,
// 
//   ///   TupleType {
//   ///     elementTypes: Type[]
//   ///   }
//   kTupleType = 15,
def TupleType : BuiltinDialectType</*enum=*/15, (type
  ArrayTypes:$elementTypes
)>;

//   ///   UnrankedMemRefType {
//   ///     shape: svarint[]
//   ///   }
//   ///
//   kUnrankedMemRefType = 16,
// 
//   ///   UnrankedMemRefTypeWithMemSpace {
//   ///     memorySpace: Attribute,
//   ///     shape: svarint[]
//   ///   }
//   /// Variant of UnrankedMemRefType with non-default memory space.
//   kUnrankedMemRefTypeWithMemSpace = 17,
// 
//   ///   UnrankedTensorType {
//   ///     elementType: Type
//   ///   }
//   ///
//   kUnrankedTensorType = 18,
def UnrankedTensorType : BuiltinDialectType</*enum=*/18, (type
  Type:$elementType
)>;

//   ///   VectorType {
//   ///     shape: svarint[],
//   ///     elementType: Type
//   ///   }
//   ///
//   kVectorType = 19,
// 
//   ///   VectorTypeWithScalableDims {
//   ///     numScalableDims: varint,
//   ///     shape: svarint[],
//   ///     elementType: Type
//   ///   }
//   /// Variant of VectorType with scalable dimensions.
//   kVectorTypeWithScalableDims = 20,
// };