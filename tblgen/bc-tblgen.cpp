//===- bc-tblgen.cpp - TableGen helper for MLIR bytecode ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <filesystem>
#include <string>

#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

cl::opt<std::string> selectedDialect("dialect",
                                     llvm::cl::desc("The dialect to gen for"));

const char *progName;

static int reportError(Twine Msg) {
  errs() << progName << ": " << Msg;
  errs().flush();
  return 1;
}

// Helper class to generate C++ bytecode parser helpers.
class Generator {
public:
  Generator(StringRef dialectName) : dialectName(dialectName) {}

  // Returns whether successfully created output dirs/files.
  void init(const std::filesystem::path &mainFileRoot);

  // Returns whether successfully terminated output files.
  void fin(raw_ostream &os);

  // Returns whether successfully emitted attribute/type parsers.
  void emitParse(StringRef kind, Record &x);

  // Returns whether successfully emitted attribute/type printers.
  void emitPrint(StringRef kind, StringRef type, ArrayRef<Record *> vec);

  // Emits parse dispatch table.
  void emitParseDispatch(StringRef kind, ArrayRef<Record *> vec);

  // Emits print dispatch table.
  void emitPrintDispatch(StringRef kind, ArrayRef<std::string> vec);

private:
  // Emits parse calls to construct givin kind.
  void emitParseHelper(StringRef kind, StringRef returnType, StringRef builder,
                       ArrayRef<Init *> args, ArrayRef<std::string> argNames,
                       StringRef failure, mlir::raw_indented_ostream &ios);

  // Emits print instructions.
  void emitPrintHelper(Record *memberRec, StringRef kind, StringRef parent,
                       StringRef name, mlir::raw_indented_ostream &ios);

  StringRef dialectName;
  std::string topStr, bottomStr;
};

void Generator::init(const std::filesystem::path &mainFileRoot) {
  // Generate top section.
  raw_string_ostream os(topStr);

  // Inject additional header include file.
  auto incHeaderFileName =
      mainFileRoot / formatv("{0}Inc.h", dialectName).str();
  if (std::filesystem::exists(incHeaderFileName)) {
    auto incFileOr =
        MemoryBuffer::getFile(incHeaderFileName.string(), /*IsText=*/true,
                              /*RequiresNullTerminator=*/false);
    if (incFileOr) {
      os << incFileOr->get()->getBuffer() << "\n";
    } else {
      PrintFatalError("FAILURE: " + incFileOr.getError().message());
    }
  }
}

void Generator::fin(raw_ostream &os) {
  os << topStr << "\n" << StringRef(bottomStr).rtrim() << "\n";
}

static std::string capitalize(StringRef str) {
  return ((Twine)toUpper(str[0]) + str.drop_front()).str();
}

std::string getCType(Record *def) {
  std::string format = "{0}";
  if (def->isSubClassOf("Array")) {
    def = def->getValueAsDef("elemT");
    format = "SmallVector<{0}>";
  }

  StringRef cType = def->getValueAsString("cType");
  if (cType.empty()) {
    if (def->isAnonymous()) {
      PrintFatalError(def->getLoc(), "Unable to determine cType");
    }

    return formatv(format.c_str(), def->getName().str());
  }
  return formatv(format.c_str(), cType.str());
}

void Generator::emitParseDispatch(StringRef kind, ArrayRef<Record *> vec) {
  raw_string_ostream sos(bottomStr);
  mlir::raw_indented_ostream os(sos);
  char const *head =
      R"(static {0} read{0}(MLIRContext* context, DialectBytecodeReader &reader))";
  os << formatv(head, capitalize(kind));
  auto funScope = os.scope(" {\n", "}\n\n");

  os << "uint64_t kind;\n";
  os << "if (failed(reader.readVarInt(kind)))\n"
     << "  return " << capitalize(kind) << "();\n";
  os << "switch (kind) ";
  {
    auto switchScope = os.scope("{\n", "}\n");
    for (auto it : vec) {
      os << formatv("case /* {0} */ {1}:\n  return read{0}(context, reader);\n",
                    it->getName(), it->getValueAsInt("enum"));
    }
    os << "default:\n"
       << "  reader.emitError() << \"unknown builtin attribute code: \" "
       << "<< kind;\n"
       << "  return " << capitalize(kind) << "();\n";
  }
  os << "return " << capitalize(kind) << "();\n";
}

void Generator::emitParse(StringRef kind, Record &attr) {
  char const *head =
      R"(static {0} read{1}(MLIRContext* context, DialectBytecodeReader &reader) )";
  raw_string_ostream os(bottomStr);
  mlir::raw_indented_ostream ios(os);
  std::string returnType = getCType(&attr);
  ios << formatv(head, returnType, attr.getName());
  DagInit *members = attr.getValueAsDag("members");
  SmallVector<std::string> argNames =
      llvm::to_vector(map_range(members->getArgNames(), [](StringInit *init) {
        return init->getAsUnquotedString();
      }));
  StringRef builder = attr.getValueAsString("cBuilder");
  emitParseHelper(kind, returnType, builder, members->getArgs(), argNames,
                  returnType + "()", ios);
}

void Generator::emitParseHelper(StringRef kind, StringRef returnType,
                                StringRef builder, ArrayRef<Init *> args,
                                ArrayRef<std::string> argNames,
                                StringRef failure,
                                mlir::raw_indented_ostream &ios) {
  auto funScope = ios.scope("{\n", "}\n\n");

  if (args.empty()) {
    ios << formatv("return {0}::get(context);\n", returnType);
    return;
  }

  // Print decls.
  std::string lastCType = "";
  for (auto [arg, name] : zip(args, argNames)) {
    DefInit *first = dyn_cast<DefInit>(arg);
    if (!first)
      PrintFatalError("Unexpected type for " + name);
    Record *def = first->getDef();

    std::string cType = getCType(def);
    if (lastCType == cType) {
      ios << ", ";
    } else {
      if (!lastCType.empty())
        ios << ";\n";
      ios << cType << " ";
    }
    ios << name;
    lastCType = cType;
  }
  ios << ";\n";

  auto listHelperName = [](StringRef name) {
    return formatv("read{0}", capitalize(name));
  };

  // Emit list helper functions.
  for (auto [arg, name] : zip(args, argNames)) {
    Record *attr = cast<DefInit>(arg)->getDef();
    if (!attr->isSubClassOf("Array"))
      continue;

    // TODO: Dedupe readers.
    Record *def = attr->getValueAsDef("elemT");
    if (!def->isSubClassOf("CompositeBytecode") &&
        (def->isSubClassOf("AttributeKind") || def->isSubClassOf("TypeKind")))
      continue;

    std::string returnType = getCType(def);
    ios << "auto " << listHelperName(name) << " = [&]() -> FailureOr<"
        << returnType << "> ";
    SmallVector<Init *> args;
    SmallVector<std::string> argNames;
    if (def->isSubClassOf("CompositeBytecode")) {
      auto members = def->getValueAsDag("members");
      args = llvm::to_vector(members->getArgs());
      argNames = llvm::to_vector(
          map_range(members->getArgNames(), [](StringInit *init) {
            return init->getAsUnquotedString();
          }));
    } else {
      args = {def->getDefInit()};
      argNames = {"temp"};
    }
    StringRef builder = def->getValueAsString("cBuilder");
    emitParseHelper(kind, returnType, builder, args, argNames, "failure()",
                    ios);
    ios << ";";
  }

  // Print parse conditional.
  {
    ios << "if ";
    auto parenScope = ios.scope("(", ") {");
    ios.indent();

    auto parsedArgs =
        llvm::to_vector(make_filter_range(args, [](Init *const attr) {
          Record *def = cast<DefInit>(attr)->getDef();
          if (def->isSubClassOf("Array"))
            return true;
          return !def->getValueAsString("cParser").empty();
        }));

    interleave(
        zip(parsedArgs, argNames),
        [&](std::tuple<llvm::Init *&, const std::string &> it) {
          Record *attr = cast<DefInit>(std::get<0>(it))->getDef();
          std::string parser;
          if (auto optParser = attr->getValueAsOptionalString("cParser")) {
            parser = *optParser;
          } else if (attr->isSubClassOf("Array")) {
            Record *def = attr->getValueAsDef("elemT");
            bool composite = def->isSubClassOf("CompositeBytecode");
            if (!composite && def->isSubClassOf("AttributeKind"))
              parser = "succeeded({0}.readAttributes({2}))";
            else if (!composite && def->isSubClassOf("TypeKind"))
              parser = "succeeded({0}.readTypes({2}))";
            else
              parser = ("succeeded({0}.readList({2}, " +
                        listHelperName(std::get<1>(it)) + "))")
                           .str();
          } else {
            PrintFatalError(attr->getLoc(), "No parser specified");
          }
          std::string type = getCType(attr);
          ios << formatv(parser.c_str(), "reader", type, std::get<1>(it));
        },
        [&]() { ios << " &&\n"; });
  }

  // Compute args to pass to create method.
  auto passedArgs = llvm::to_vector(make_filter_range(
      argNames, [](StringRef str) { return !str.starts_with("_"); }));
  std::string argStr;
  raw_string_ostream argStream(argStr);
  interleaveComma(passedArgs, argStream,
                  [&](const std::string &str) { argStream << str; });
  // Return the invoked constructor.
  ios << "\nreturn "
      << formatv(builder.str().c_str(), returnType, argStream.str()) << ";\n";
  ios.unindent();

  // TODO: Emit error in debug.
  // This assumes the result types in error case can always be empty
  // constructed. ios << "}\nreturn mlirBytecodeEmitError(\"invalid " <<
  // attr.getName()
  //     << "\");\n";
  ios << "}\nreturn " << failure << ";\n";
}

void Generator::emitPrint(StringRef kind, StringRef type,
                          ArrayRef<Record *> vec) {
  char const *head =
      R"(static void write({0} {1}, DialectBytecodeWriter &writer) )";
  raw_string_ostream os(bottomStr);
  mlir::raw_indented_ostream ios(os);
  ios << formatv(head, type, kind);
  auto funScope = ios.scope("{\n", "}\n\n");

  // Check that predicates specified if multiple bytecode instances.
  for (Record *rec : vec) {
    StringRef pred = rec->getValueAsString("printerPredicate");
    if (vec.size() > 1 && pred.empty()) {
      for (Record *rec : vec) {
        StringRef pred = rec->getValueAsString("printerPredicate");
        if (vec.size() > 1 && pred.empty())
          PrintError(rec->getLoc(),
                     "Requires parsing predicate given common cType");
      }
      PrintFatalError("Unspecified for shared cType " + type);
    }
  }

  for (Record *rec : vec) {
    StringRef pred = rec->getValueAsString("printerPredicate");
    if (!pred.empty()) {
      ios << "if (" << formatv(pred.str().c_str(), kind) << ") {\n";
      ios.indent();
    }

    ios << "writer.writeVarInt(/* " << rec->getName() << " */ "
        << rec->getValueAsInt("enum") << ");\n";

    auto members = rec->getValueAsDag("members");
    for (auto [arg, name] :
         llvm::zip(members->getArgs(), members->getArgNames())) {
      DefInit *def = dyn_cast<DefInit>(arg);
      assert(def);
      Record *memberRec = def->getDef();
      emitPrintHelper(memberRec, kind, kind, name->getAsUnquotedString(), ios);
    }

    if (!pred.empty()) {
      ios.unindent();
      ios << "}\n";
    }
  }
}

void Generator::emitPrintHelper(Record *memberRec, StringRef kind,
                                StringRef parent, StringRef name,
                                mlir::raw_indented_ostream &ios) {
  std::string getter;
  if (auto cGetter = memberRec->getValueAsOptionalString("cGetter");
      cGetter && !cGetter->empty()) {
    getter = formatv(cGetter->str().c_str(), parent,
                     "get" + convertToCamelFromSnakeCase(name, true));
  } else {
    getter =
        formatv("{0}.get{1}()", parent, convertToCamelFromSnakeCase(name, true))
            .str();
  }

  if (memberRec->isSubClassOf("Array")) {
    Record *def = memberRec->getValueAsDef("elemT");
    if (!def->isSubClassOf("CompositeBytecode")) {
      if (def->isSubClassOf("AttributeKind")) {
        ios << "writer.writeAttributes(" << getter << ");\n";
        return;
      }
      if (def->isSubClassOf("TypeKind")) {
        ios << "writer.writeTypes(" << getter << ");\n";
        return;
      }
    }
    std::string returnType = getCType(def);
    ios << "writer.writeList(" << getter << ", [&](" << returnType << " "
        << kind << ") ";
    auto lambdaScope = ios.scope("{\n", "});\n");
    return emitPrintHelper(def, kind, kind, kind, ios);
  }
  if (memberRec->isSubClassOf("CompositeBytecode")) {
    auto members = memberRec->getValueAsDag("members");
    for (auto [arg, argName] :
         zip(members->getArgs(), members->getArgNames())) {
      DefInit *def = dyn_cast<DefInit>(arg);
      assert(def);
      emitPrintHelper(def->getDef(), kind, parent,
                      argName->getAsUnquotedString(), ios);
    }
  }

  if (std::string printer = memberRec->getValueAsString("cPrinter").str();
      !printer.empty())
    ios << formatv(printer.c_str(), "writer", kind, getter) << ";\n";
}

void Generator::emitPrintDispatch(StringRef kind, ArrayRef<std::string> vec) {
  raw_string_ostream sos(bottomStr);
  mlir::raw_indented_ostream os(sos);
  char const *head = R"(static LogicalResult write{0}({0} {1},
                                DialectBytecodeWriter &writer))";
  os << formatv(head, capitalize(kind), kind);
  auto funScope = os.scope(" {\n", "}\n\n");

  os << "return TypeSwitch<" << capitalize(kind) << ", LogicalResult>(" << kind
     << ")";
  auto switchScope = os.scope("", "");
  for (StringRef type : vec) {
    os << "\n.Case([&](" << type << " t)";
    auto caseScope = os.scope(" {\n", "})");
    os << "return write(t, writer), success();\n";
  }
  os << "\n.Default([&](" << capitalize(kind) << ") { return failure(); });\n";
}

struct AttrOrType {
  std::vector<Record *> attr, type;
};

static bool tableGenMain(raw_ostream &os, RecordKeeper &records) {
  MapVector<StringRef, AttrOrType> dialectAttrOrType;
  Record *attr = records.getClass("DialectAttribute");
  Record *type = records.getClass("DialectType");
  for (auto &it : records.getAllDerivedDefinitions("DialectAttrOrType")) {
    if (!selectedDialect.empty() &&
        it->getValueAsString("dialect") != selectedDialect)
      continue;

    if (it->isSubClassOf(attr)) {
      dialectAttrOrType[it->getValueAsString("dialect")].attr.push_back(it);
    } else if (it->isSubClassOf(type)) {
      dialectAttrOrType[it->getValueAsString("dialect")].type.push_back(it);
    }
  }

  if (dialectAttrOrType.size() != 1)
    return reportError("Single dialect per invocation required (either only "
                       "one in input file or specified via dialect option)");

  // Compare two records by enum value.
  auto compEnum = [](Record *lhs, Record *rhs) -> int {
    return lhs->getValueAsInt("enum") < rhs->getValueAsInt("enum");
  };

  auto mainFile =
      std::filesystem::path(records.getInputFilename()).remove_filename();

  auto it = dialectAttrOrType.front();
  Generator gen(it.first);
  gen.init(mainFile);

  SmallVector<std::vector<Record *> *, 2> vecs;
  SmallVector<std::string, 2> kinds;
  vecs.push_back(&it.second.attr);
  kinds.push_back("attribute");
  vecs.push_back(&it.second.type);
  kinds.push_back("type");
  for (auto [vec, kind] : zip(vecs, kinds)) {
    // Handle Attribute emission.
    std::sort(vec->begin(), vec->end(), compEnum);
    std::map<std::string, std::vector<Record *>> perType;
    for (auto kt : *vec)
      perType[getCType(kt)].push_back(kt);
    for (auto jt : perType) {
      for (auto kt : jt.second)
        gen.emitParse(kind, *kt);
      gen.emitPrint(kind, jt.first, jt.second);
    }
    gen.emitParseDispatch(kind, *vec);

    SmallVector<std::string> types;
    for (auto it : perType) {
      types.push_back(it.first);
    }
    gen.emitPrintDispatch(kind, types);
  }
  gen.fin(os);

  return false;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(argc, argv);
  progName = argv[0];

  return TableGenMain(argv[0], &tableGenMain);
}