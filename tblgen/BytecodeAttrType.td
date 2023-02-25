// Bytecode base classes/defs.
// Helper classes/defs to help build a reader/writer.

class Bytecode<string parse, string build, string print, string t=""> {
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
  // {1} == value being printed
  string cPrinter = print;

  // Template for getter from in memory form.
  // TODO: string cGetter

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

def EmptyBytecode : Bytecode<"", "", "", "">;
class WithParser<string p="", Bytecode t=EmptyBytecode> :
  Bytecode<p, t.cBuilder, t.cPrinter, t.cType>;
class WithBuilder<string b="", Bytecode t=EmptyBytecode> :
  Bytecode<t.cParser, b, t.cPrinter, t.cType>;
class WithPrinter<string p="", Bytecode t=EmptyBytecode> :
  Bytecode<t.cParser, t.cBuilder, p, t.cType>;
class WithType<string ty="", Bytecode t=EmptyBytecode> :
  Bytecode<t.cParser, t.cBuilder, t.cPrinter, ty>;

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

class KnownWidthAPInt<string s> :
  WithParser <"succeeded(readAPIntWithKnownWidth({0}, " # s # ", {2}))",
  WithBuilder<"{1}",
  WithPrinter<"{0}.writeAPIntWithKnownWidth({2})",
  WithType   <"APInt">>>>;

// Helper to define variable that is defined later but not parsed nor printed.
class LocalVar<string t, string d> : Bytecode<
  "(({2} = " # d # "), true)", "{1}", "", t>;

// Array instances.
class Array<Bytecode t> {
  Bytecode elemT = t;

  string cBuilder = "{1}";
}

// Define dialect attribute or type.
class DialectAttrOrType<string di, int e, dag d> {
  int enum = e;
  // Any members starting with underscore is not fed to create function but
  // treated as purely local variable.
  dag members = d;
  string dialect = di;

  // When needing to specify a custom return type.
  string cType = "";

  // Any post-processing that needs to be done.
  code postProcess = "";
}

class DialectAttribute<string di, int e, dag d> : DialectAttrOrType<di, e, d>,
    AttributeKind {
  let cParser = "succeeded({0}.readAttribute<{1}>({2}))";
  let cBuilder = "{0}::get({1})";
}
class DialectType<string di, int e, dag d> : DialectAttrOrType<di, e, d>,
    TypeKind {
  let cParser = "succeeded({0}.readAttribute<{1}>({2}))";
  let cBuilder = "{0}::get({1})";
}

def attr;
def type;

// ----------------
// Builtin dialect.
// ================

class BuiltinDialectAttribute<int e, dag d> : DialectAttribute<"Builtin", e, d>;
class BuiltinDialectType<int e, dag d> : DialectType<"Builtin", e, d>;

def LocationAttr : AttributeKind {
  let cBuilder = "{0}::get({1})";
  let cType = "LocationAttr";
}

def Location : CompositeBytecode {
  dag members = (attr
    LocationAttr:$value
  );
  let cBuilder = "Location({1})";
}

def String :
  WithParser <"succeeded({0}.readString({2}))",
  WithBuilder<"{1}",
  WithPrinter<"{0}.writeOwnedString({1})",
  WithType   <"StringRef">>>>;

// enum AttributeCode {
//  ///   ArrayAttr {
//  ///     elements: Attribute[]
//  ///   }
//  ///
//  kArrayAttr = 0,
//

let cType = "StringAttr" in {
//  ///   StringAttr {
//  ///     value: string
//  ///   }
//  kStringAttr = 2,
def StringAttr : BuiltinDialectAttribute</*enum=*/2, (attr
  String:$value
)> { let printerPredicate = "{0}.getType().isa<NoneType>()"; }

//  ///   StringAttrWithType {
//  ///     value: string,
//  ///     type: Type
//  ///   }
//  /// A variant of StringAttr with a type.
//  kStringAttrWithType = 3,
def StringAttrWithType : BuiltinDialectAttribute</*enum=*/3, (attr
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
def DictionaryAttr : BuiltinDialectAttribute</*enum=*/1, (attr
  // TODO: handle context constructors a bit nicer.
  Bytecode<"({2}=context)", "{1}", "", "MLIRContext*">:$mlirContext,

  Array<NamedAttribute>:$value
)>;

let cType = "SymbolRefAttr" in {
//  ///   FlatSymbolRefAttr {
//  ///     rootReference: StringAttr
//  ///   }
//  /// A variant of SymbolRefAttr with no leaf references.
//  kFlatSymbolRefAttr = 4,
def FlatSymbolRefAttr: BuiltinDialectAttribute</*enum=*/4, (attr
  StringAttr:$rootReference
)> { let printerPredicate = "{0}.getNestedReferences().empty()"; }

//  ///   SymbolRefAttr {
//  ///     rootReference: StringAttr,
//  ///     leafReferences: FlatSymbolRefAttr[]
//  ///   }
//  kSymbolRefAttr = 5,
def SymbolRefAttr: BuiltinDialectAttribute</*enum=*/5, (attr
  StringAttr:$rootReference,
  Array<FlatSymbolRefAttr>:$nestedReferences
)> { let printerPredicate = "!{0}.getNestedReferences().empty()"; }
}

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
  LocalVar<"unsigned", "getIntegerBitWidth({0}, type)">:$_width,
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
def CallSiteLoc : BuiltinDialectAttribute</*enum=*/10, (attr
  LocationAttr:$callee,
  LocationAttr:$caller
)>;

//  ///   FileLineColLoc {
//  ///     filename: StringAttr,
//  ///     line: varint,
//  ///     column: varint
//  ///   }
//  kFileLineColLoc = 11,
def FileLineColLoc : BuiltinDialectAttribute</*enum=*/11, (attr
  StringAttr:$filename,
  VarInt:$line,
  VarInt:$column
)>;

let cType = "FusedLoc" in {
//  ///   FusedLoc {
//  ///     locations: LocationAttr[]
//  ///   }
//  kFusedLoc = 12,
def FusedLoc : BuiltinDialectAttribute</*enum=*/12, (attr
  // TODO: handle context constructors a bit nicer.
  Bytecode<"({2}=context)", "{1}", "", "MLIRContext*">:$mlirContext,

  Array<Location>:$locations
)> { let printerPredicate = "!{0}.getMetadata()"; }

//  ///   FusedLocWithMetadata {
//  ///     locations: LocationAttr[],
//  ///     metadata: Attribute
//  ///   }
//  /// A variant of FusedLoc with metadata.
//  kFusedLocWithMetadata = 13,
def FusedLocWithMetadata : BuiltinDialectAttribute</*enum=*/13, (attr
  Array<Location>:$locations,
  Attribute:$metadata
)> { let printerPredicate = "{0}.getMetadata()"; }
}

//  ///   NameLoc {
//  ///     name: StringAttr,
//  ///     childLoc: LocationAttr
//  ///   }
//  kNameLoc = 14,
def NameLoc : BuiltinDialectAttribute</*enum=*/14, (attr
  StringAttr:$name,
  LocationAttr:$childLoc
)>;

//  ///   UnknownLoc {
//  ///   }
//  kUnknownLoc = 15,
def UnknownLoc : BuiltinDialectAttribute</*enum=*/15, (attr)>;

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
  // TODO: handle context constructors a bit nicer.
  Bytecode<"({2}=context)", "{1}", "", "MLIRContext*">:$mlirContext,
  // Yes not pretty, 
  Bytecode<
    "succeeded({0}.readVarInt({2}))",
    "{1}",
    "{0}.writeVarInt({2}.getWidth() << 2 | {2}.getSignedness())",
    "uint64_t">:$_widthAndSignedness,
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
def IndexType : BuiltinDialectType</*enum=*/1, (type)>;

//   ///   FunctionType {
//   ///     inputs: Type[],
//   ///     results: Type[]
//   ///   }
//   ///
//   kFunctionType = 2,
def FunctionType : BuiltinDialectType</*enum=*/2, (type
  // TODO: handle context constructors a bit nicer.
  Bytecode<"({2}=context)", "{1}", "", "MLIRContext*">:$mlirContext,

  Array<Type>:$inputs,
  Array<Type>:$results
)>;

//   ///   BFloat16Type {
//   ///   }
//   ///
//   kBFloat16Type = 3,
def BFloat16Type : BuiltinDialectType</*enum=*/3, (type)>;

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
  // TODO: handle context constructors a bit nicer.
  Bytecode<"({2}=context)", "{1}", "", "MLIRContext*">:$mlirContext,

  Array<Type>:$types
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
