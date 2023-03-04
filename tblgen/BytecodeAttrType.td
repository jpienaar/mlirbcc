// Bytecode base classes/defs.
// Helper classes/defs to help build a reader/writer.

class Bytecode<string parse="", string build="", string print="", string get="", string t=""> {
  // Template for parsing.
  // {0} == dialect bytecode reader
  // {1} == result type of parsed instance
  // {2} == variable being parsed
  // If parser is not specified, then the parse of members is used.
  string cParser = parse;

  // Template for building from parsed.
  // {0} == result type of parsed instance
  // {1} == args/members comma separated
  string cBuilder = build;

  // Template for printing.
  // {0} == dialect bytecode writer
  // {1} == parent attribute/type name
  // {2} == getter
  string cPrinter = print;

  // Template for getter from in memory form.
  // {0} == attribute/type
  // {1} == member instance
  // {2} == get + UpperCamelFromSnake({1})
  string cGetter = get;

  // Type built.
  // Note: if cType is empty, then name of def is used.
  string cType = t;

  // Predicate guarding parse method as an Attribute/Type could have multiple
  // parse methods, specify predicates to be orthogonal and cover entire
  // "print space" to avoid order dependence.
  // If empty then method is unconditional.
  // {0} == predicate function to apply on value dyn_casted to cType.
  string printerPredicate = "";
}

class WithParser<string p="", Bytecode t=Bytecode<>> :
  Bytecode<p, t.cBuilder, t.cPrinter, t.cGetter, t.cType>;
class WithBuilder<string b="", Bytecode t=Bytecode<>> :
  Bytecode<t.cParser, b, t.cPrinter, t.cGetter, t.cType>;
class WithPrinter<string p="", Bytecode t=Bytecode<>> :
  Bytecode<t.cParser, t.cBuilder, p, t.cGetter, t.cType>;
class WithType<string ty="", Bytecode t=Bytecode<>> :
  Bytecode<t.cParser, t.cBuilder, t.cPrinter, t.cGetter, ty>;
class WithGetter<string g="", Bytecode t=Bytecode<>> :
  Bytecode<t.cParser, t.cBuilder, t.cPrinter, g, t.cType>;

class CompositeBytecode<string t = ""> : WithType<t>;

class AttributeKind :
  WithParser <"succeeded({0}.readAttribute({2}))",
  WithBuilder<"{1}",
  WithPrinter<"{0}.writeAttribute({2})">>>;
def Attribute : AttributeKind;
class TypeKind :
  WithParser <"succeeded({0}.readType({2}))",
  WithBuilder<"{1}",
  WithPrinter<"{0}.writeType({2})">>>;
def Type : TypeKind;
def VarInt :
  WithParser <"succeeded({0}.readVarInt({2}))",
  WithBuilder<"{1}",
  WithPrinter<"{0}.writeVarInt({2})",
  WithType   <"uint64_t">>>>;
def SignedVarInt :
  WithParser <"succeeded({0}.readSignedVarInt({2}))",
  WithBuilder<"{1}",
  WithPrinter<"{0}.writeSignedVarInt({2})",
  WithGetter<"{0}",
  WithType   <"int64_t">>>>>;
def Blob :
  WithParser <"succeeded({0}.readBlob({2}))",
  WithBuilder<"{1}",
  WithPrinter<"{0}.writeOwnedBlob({2})",
  WithType   <"ArrayRef<char>">>>>;

class KnownWidthAPInt<string s> :
  WithParser <"succeeded(readAPIntWithKnownWidth({0}, " # s # ", {2}))",
  WithBuilder<"{1}",
  WithPrinter<"{0}.writeAPIntWithKnownWidth({2})",
  WithType   <"APInt">>>>;
class KnownSemanticsAPFloat<string s> :
  WithParser <"succeeded(readAPFloatWithKnownSemantics({0}, " # s # ", *{2}))",
  WithBuilder<"{1}",
  WithPrinter<"{0}.writeAPFloatWithKnownSemantics({2})",
  WithType   <"std::optional<APFloat>">>>>;
class ResourceHandle<string s> :
  WithParser <"succeeded(readResourceHandle<" # s # ">({0}, *{2}))",
  WithBuilder<"{1}",
  WithPrinter<"{0}.writeResourceHandle({2})",
  WithType   <"std::optional<" # s # ">">>>>;

// Helper to define variable that is defined later but not parsed nor printed.
class LocalVar<string t, string d> :
  WithParser <"(({2} = " # d # "), true)",
  WithBuilder<"{1}",
  WithPrinter<"",
  WithType   <t>>>>;

// Array instances.
class Array<Bytecode t> {
  Bytecode elemT = t;

  string cBuilder = "{1}";
}

// Define dialect attribute or type.
class DialectAttrOrType<string di, dag d> {
  // Any members starting with underscore is not fed to create function but
  // treated as purely local variable.
  dag members = d;
  string dialect = di;

  // When needing to specify a custom return type.
  string cType = "";

  // Any post-processing that needs to be done.
  code postProcess = "";
}

class DialectAttribute<string di, dag d> : DialectAttrOrType<di, d>,
    AttributeKind {
  let cParser = "succeeded({0}.readAttribute<{1}>({2}))";
  let cBuilder = "get<{0}>(context, {1})";
}
class DialectType<string di, dag d> : DialectAttrOrType<di, d>,
    TypeKind {
  let cParser = "succeeded({0}.readType<{1}>({2}))";
  let cBuilder = "get<{0}>(context, {1})";
}

class DialectAttributes<string name> {
  string dialect = name;
  list<DialectAttrOrType> elems;
}

class DialectTypes<string name> {
  string dialect = name;
  list<DialectAttrOrType> elems;
}

def attr;
def type;

// ----------------
// Builtin dialect.
// ================

class BuiltinDialectAttribute<dag d> : DialectAttribute<"Builtin", d>;
class BuiltinDialectType<dag d> : DialectType<"Builtin", d>;

def LocationAttr : AttributeKind;

def Location : CompositeBytecode {
  dag members = (attr
    WithGetter<"(LocationAttr){0}", WithType<"LocationAttr", LocationAttr>>:$value
  );
  let cBuilder = "Location({1})";
}

def String :
  WithParser <"succeeded({0}.readString({2}))",
  WithBuilder<"{1}",
  WithPrinter<"{0}.writeOwnedString({2})",
  WithGetter <"{0}",
  WithType   <"StringRef">>>>>;

// enum AttributeCode {
//  ///   ArrayAttr {
//  ///     elements: Attribute[]
//  ///   }
//  ///
//  kArrayAttr = 0,
//
def ArrayAttr : BuiltinDialectAttribute<(attr
  Array<Attribute>:$value
)>;

let cType = "StringAttr" in {
//  ///   StringAttr {
//  ///     value: string
//  ///   }
//  kStringAttr = 2,
def StringAttr : BuiltinDialectAttribute<(attr
  String:$value
)> {
  let printerPredicate = "{0}.getType().isa<NoneType>()";
}

//  ///   StringAttrWithType {
//  ///     value: string,
//  ///     type: Type
//  ///   }
//  /// A variant of StringAttr with a type.
//  kStringAttrWithType = 3,
def StringAttrWithType : BuiltinDialectAttribute<(attr
  String:$value,
  Type:$type
)> { let printerPredicate = "!{0}.getType().isa<NoneType>()"; }
}

//  ///   DictionaryAttr {
//  ///     attrs: <StringAttr, Attribute>[]
//  ///   }
//  kDictionaryAttr = 1,
def NamedAttribute : CompositeBytecode {
  dag members = (attr
    StringAttr:$name,
    Attribute:$value
  );
  let cBuilder = "NamedAttribute({1})";
}
def DictionaryAttr : BuiltinDialectAttribute<(attr
  Array<NamedAttribute>:$value
)>;

//  ///   FlatSymbolRefAttr {
//  ///     rootReference: StringAttr
//  ///   }
//  /// A variant of SymbolRefAttr with no leaf references.
//  kFlatSymbolRefAttr = 4,
def FlatSymbolRefAttr: BuiltinDialectAttribute<(attr
  StringAttr:$rootReference
)>;

//  ///   SymbolRefAttr {
//  ///     rootReference: StringAttr,
//  ///     leafReferences: FlatSymbolRefAttr[]
//  ///   }
//  kSymbolRefAttr = 5,
def SymbolRefAttr: BuiltinDialectAttribute<(attr
  StringAttr:$rootReference,
  Array<FlatSymbolRefAttr>:$nestedReferences
)>;

//  ///   TypeAttr {
//  ///     value: Type
//  ///   }
//  kTypeAttr = 6,
def TypeAttr: BuiltinDialectAttribute<(attr
  Type:$value
)>;

//  ///   UnitAttr {
//  ///   }
//  kUnitAttr = 7,
def UnitAttr: BuiltinDialectAttribute<(attr)>;

//  ///   IntegerAttr {
//  ///     type: Type
//  ///     value: APInt,
//  ///   }
//  kIntegerAttr = 8,
def IntegerAttr: BuiltinDialectAttribute<(attr
  Type:$type,
  KnownWidthAPInt<"type">:$value
)>;

//
//  ///   FloatAttr {
//  ///     type: FloatType
//  ///     value: APFloat
//  ///   }
//  kFloatAttr = 9,
defvar FloatType = Type;
def FloatAttr : BuiltinDialectAttribute<(attr
  FloatType:$type,
  KnownSemanticsAPFloat<"type">:$value
)> {
  let cBuilder = "get<{0}>(context, type, *value)";
}

//  ///   CallSiteLoc {
//  ///    callee: LocationAttr,
//  ///    caller: LocationAttr
//  ///   }
//  kCallSiteLoc = 10,
def CallSiteLoc : BuiltinDialectAttribute<(attr
  LocationAttr:$callee,
  LocationAttr:$caller
)>;

//  ///   FileLineColLoc {
//  ///     filename: StringAttr,
//  ///     line: varint,
//  ///     column: varint
//  ///   }
//  kFileLineColLoc = 11,
def FileLineColLoc : BuiltinDialectAttribute<(attr
  StringAttr:$filename,
  VarInt:$line,
  VarInt:$column
)>;

let cType = "FusedLoc",
    cBuilder = "cast<FusedLoc>(get<FusedLoc>(context, {1}))" in {
//  ///   FusedLoc {
//  ///     locations: Location[]
//  ///   }
//  kFusedLoc = 12,
def FusedLoc : BuiltinDialectAttribute<(attr
  Array<Location>:$locations
)> {
  let printerPredicate = "!{0}.getMetadata()";
}

//  ///   FusedLocWithMetadata {
//  ///     locations: LocationAttr[],
//  ///     metadata: Attribute
//  ///   }
//  /// A variant of FusedLoc with metadata.
//  kFusedLocWithMetadata = 13,
def FusedLocWithMetadata : BuiltinDialectAttribute<(attr
  Array<Location>:$locations,
  Attribute:$metadata
)> {
  let printerPredicate = "{0}.getMetadata()";
}
}

//  ///   NameLoc {
//  ///     name: StringAttr,
//  ///     childLoc: LocationAttr
//  ///   }
//  kNameLoc = 14,
def NameLoc : BuiltinDialectAttribute<(attr
  StringAttr:$name,
  LocationAttr:$childLoc
)>;

//  ///   UnknownLoc {
//  ///   }
//  kUnknownLoc = 15,
def UnknownLoc : BuiltinDialectAttribute<(attr)>;

//  ///   DenseResourceElementsAttr {
//  ///     type: Type,
//  ///     handle: ResourceHandle
//  ///   }
//  kDenseResourceElementsAttr = 16,
def DenseResourceElementsAttr : BuiltinDialectAttribute<(attr
  Type:$type,
  ResourceHandle<"DenseResourceElementsHandle">:$rawHandle
)> {
  // Note: order of serialization does not match order of builder.
  let cBuilder = "get<{0}>(context, type, *rawHandle)";
}

let cType = "RankedTensorType" in {
//   ///   RankedTensorType {
//   ///     shape: svarint[],
//   ///     elementType: Type,
//   ///   }
//   ///
//   kRankedTensorType = 13,
def RankedTensorType : BuiltinDialectType<(type
  Array<SignedVarInt>:$shape,
  Type:$elementType
)> {
  let printerPredicate = "!{0}.getEncoding()";
}

//   ///   RankedTensorTypeWithEncoding {
//   ///     encoding: Attribute,
//   ///     shape: svarint[],
//   ///     elementType: Type
//   ///   }
//   /// Variant of RankedTensorType with an encoding.
//   kRankedTensorTypeWithEncoding = 14,
def RankedTensorTypeWithEncoding : BuiltinDialectType<(type
  Attribute:$encoding,
  Array<SignedVarInt>:$shape,
  Type:$elementType
)> {
  let printerPredicate = "{0}.getEncoding()";
  // Note: order of serialization does not match order of builder.
  let cBuilder = "get<{0}>(context, shape, elementType, encoding)";
}
}

//  ///   DenseArrayAttr {
//  ///     type: RankedTensorType,
//  ///     data: blob
//  ///   }
//  kDenseArrayAttr = 17,
def DenseArrayAttr : BuiltinDialectAttribute<(attr
  RankedTensorType:$elementType,
  VarInt:$size,
  Blob:$rawData
)>;

//  ///   DenseIntOrFPElementsAttr {
//  ///     type: ShapedType,
//  ///     data: blob
//  ///   }
//  kDenseIntOrFPElementsAttr = 18,
defvar ShapedType = Type;
def DenseElementsAttr : WithType<"DenseIntElementsAttr", Attribute>;
def DenseIntOrFPElementsAttr : BuiltinDialectAttribute<(attr
  ShapedType:$type,
  Blob:$rawData
)> {
  let cBuilder = "cast<{0}>({0}::getFromRawBuffer({1}))";
}

//  ///   DenseStringElementsAttr {
//  ///     type: ShapedType,
//  ///     isSplat: varint,
//  ///     data: string[]
//  ///   }
//  kDenseStringElementsAttr = 19,
def DenseStringElementsAttr : BuiltinDialectAttribute<(attr
  ShapedType:$type,
  // TODO: Wrong
  WithGetter<"{0}.isSplat()", VarInt>:$_isSplat,
  Array<String>:$rawStringData
)>;

//  ///   SparseElementsAttr {
//  ///     type: ShapedType,
//  ///     indices: DenseIntElementsAttr,
//  ///     values: DenseElementsAttr
//  ///   }
//  kSparseElementsAttr = 20,
def DenseIntElementsAttr : WithType<"DenseIntElementsAttr", Attribute>;
def SparseElementsAttr : BuiltinDialectAttribute<(attr
  ShapedType:$type,
  DenseIntElementsAttr:$indices,
  DenseElementsAttr:$values
)>;

// Types
// -----

// enum TypeCode {
//   ///   IntegerType {
//   ///     widthAndSignedness: varint // (width << 2) | (signedness)
//   ///   }
//   ///
//   kIntegerType = 0,
def IntegerType : BuiltinDialectType<(type
  // Yes not pretty, 
  WithParser<"succeeded({0}.readVarInt({2}))",
    WithBuilder<"{1}",
    WithPrinter<"{0}.writeVarInt({1}.getWidth() << 2 | {1}.getSignedness())",
    WithType   <"uint64_t">>>>:$_widthAndSignedness,
  // Split up parsed varint for create method.
  LocalVar<"uint64_t", "_widthAndSignedness >> 2">:$width,
  LocalVar<"IntegerType::SignednessSemantics",
    "static_cast<IntegerType::SignednessSemantics>(_widthAndSignedness & 0x3)">:$signedness
)>;

// 
//   ///   IndexType {
//   ///   }
//   ///
//   kIndexType = 1,
def IndexType : BuiltinDialectType<(type)>;

//   ///   FunctionType {
//   ///     inputs: Type[],
//   ///     results: Type[]
//   ///   }
//   ///
//   kFunctionType = 2,
def FunctionType : BuiltinDialectType<(type
  Array<Type>:$inputs,
  Array<Type>:$results
)>;

//   ///   BFloat16Type {
//   ///   }
//   ///
//   kBFloat16Type = 3,
def BFloat16Type : BuiltinDialectType<(type)>;

//   ///   Float16Type {
//   ///   }
//   ///
//   kFloat16Type = 4,
def Float16Type : BuiltinDialectType<(type)>;

//   ///   Float32Type {
//   ///   }
//   ///
//   kFloat32Type = 5,
def Float32Type : BuiltinDialectType<(type)>;

//   ///   Float64Type {
//   ///   }
//   ///
//   kFloat64Type = 6,
def Float64Type : BuiltinDialectType<(type)>;

//   ///   Float80Type {
//   ///   }
//   ///
//   kFloat80Type = 7,
def Float80Type : BuiltinDialectType<(type)>;

//   ///   Float128Type {
//   ///   }
//   ///
//   kFloat128Type = 8,
def Float128Type : BuiltinDialectType<(type)>;

//   ///   ComplexType {
//   ///     elementType: Type
//   ///   }
//   ///
//   kComplexType = 9,
def ComplexType : BuiltinDialectType<(type
  Type:$elementType
)>;

let cType = "MemRefType" in {
//   ///   MemRefType {
//   ///     shape: svarint[],
//   ///     elementType: Type,
//   ///     layout: Attribute
//   ///   }
//   ///
//   kMemRefType = 10,
def MemRefType : BuiltinDialectType<(type
  Array<SignedVarInt>:$shape,
  Type:$elementType,
  Attribute:$layout
)> {
  let printerPredicate = "!{0}.getMemorySpace()";
}

//   ///   MemRefTypeWithMemSpace {
//   ///     memorySpace: Attribute,
//   ///     shape: svarint[],
//   ///     elementType: Type,
//   ///     layout: Attribute
//   ///   }
//   /// Variant of MemRefType with non-default memory space.
//   kMemRefTypeWithMemSpace = 11,
def MemRefTypeWithMemSpace : BuiltinDialectType<(type
  Attribute:$memorySpace,
  Array<SignedVarInt>:$shape,
  Type:$elementType,
  Attribute:$layout
)> {
  let printerPredicate = "!!{0}.getMemorySpace()";
  // Note: order of serialization does not match order of builder.
  let cBuilder = "get<{0}>(context, shape, elementType, layout, memorySpace)";
}
}

//   ///   NoneType {
//   ///   }
//   ///
//   kNoneType = 12,
def NoneType : BuiltinDialectType<(type)>;

//   ///   TupleType {
//   ///     elementTypes: Type[]
//   ///   }
//   kTupleType = 15,
def TupleType : BuiltinDialectType<(type
  Array<Type>:$types
)>;

let cType = "UnrankedMemRefType" in {
//   ///   UnrankedMemRefType {
//   ///     elementType: Type
//   ///   }
//   ///
//   kUnrankedMemRefType = 16,
def UnrankedMemRefType : BuiltinDialectType<(type
  Type:$elementType
)> {
  let printerPredicate = "!{0}.getMemorySpace()";
  let cBuilder = "get<{0}>(context, elementType, Attribute())";
}

//   ///   UnrankedMemRefTypeWithMemSpace {
//   ///     memorySpace: Attribute,
//   ///     elementType: Type
//   ///   }
//   /// Variant of UnrankedMemRefType with non-default memory space.
//   kUnrankedMemRefTypeWithMemSpace = 17,
def UnrankedMemRefTypeWithMemSpace : BuiltinDialectType<(type
  Attribute:$memorySpace,
  Type:$elementType
)> {
  let printerPredicate = "{0}.getMemorySpace()";
  // Note: order of serialization does not match order of builder.
  let cBuilder = "get<{0}>(context, elementType, memorySpace)";
}
}

//   ///   UnrankedTensorType {
//   ///     elementType: Type
//   ///   }
//   ///
//   kUnrankedTensorType = 18,
def UnrankedTensorType : BuiltinDialectType<(type
  Type:$elementType
)>;

let cType = "VectorType" in {
//   ///   VectorType {
//   ///     shape: svarint[],
//   ///     elementType: Type
//   ///   }
//   ///
//   kVectorType = 19,
def VectorType : BuiltinDialectType<(type
  Array<SignedVarInt>:$shape,
  Type:$elementType
)> {
  let printerPredicate = "!{0}.getNumScalableDims()";
}

//   ///   VectorTypeWithScalableDims {
//   ///     numScalableDims: varint,
//   ///     shape: svarint[],
//   ///     elementType: Type
//   ///   }
//   /// Variant of VectorType with scalable dimensions.
//   kVectorTypeWithScalableDims = 20,
def VectorTypeWithScalableDims : BuiltinDialectType<(type
  VarInt:$numScalableDims,
  Array<SignedVarInt>:$shape,
  Type:$elementType
)> {
  let printerPredicate = "!{0}.getNumScalableDims()";
  // Note: order of serialization does not match order of builder.
  let cBuilder = "get<{0}>(context, shape, elementType, numScalableDims)";
}
}

def BuiltinDialectAttributes : DialectAttributes<"Builtin"> {
  let elems = [
    ArrayAttr,
    DictionaryAttr,
    StringAttr,
    StringAttrWithType,
    FlatSymbolRefAttr,
    SymbolRefAttr,
    TypeAttr,
    UnitAttr,
    IntegerAttr,
    FloatAttr,
    CallSiteLoc,
    FileLineColLoc,
    FusedLoc,
    FusedLocWithMetadata,
    NameLoc,
    UnknownLoc,
    DenseResourceElementsAttr,
    DenseArrayAttr,
    DenseIntOrFPElementsAttr,
    DenseStringElementsAttr,
    SparseElementsAttr
  ];
}

def BuiltinDialectTypes : DialectTypes<"Builtin"> {
  let elems = [
    IntegerType,
    IndexType,
    FunctionType,
    BFloat16Type,
    Float16Type,
    Float32Type,
    Float64Type,
    Float80Type,
    Float128Type,
    ComplexType,
    MemRefType,
    MemRefTypeWithMemSpace,
    NoneType,
    RankedTensorType,
    RankedTensorTypeWithEncoding,
    TupleType,
    UnrankedMemRefType,
    UnrankedMemRefTypeWithMemSpace,
    UnrankedTensorType,
    VectorType,
    VectorTypeWithScalableDims
  ];
}