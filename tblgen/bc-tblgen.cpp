
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

cl::opt<bool> generateWeak{
    "emit-weak", cl::desc("Emit weak version of extern functions too"),
    cl::init(false)};

const char *progName;

static int reportError(Twine Msg) {
  errs() << progName << ": " << Msg;
  errs().flush();
  return 1;
}

const char licenseHeader[] =
    R"(//===-- {0}.c.inc - Parser driver for {0} dialect ---*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// {0} dialect Attribute & Type parser driver.
//===----------------------------------------------------------------------===//

)";

// Helper class to generate C++ bytecode parser helpers.
class Generator {
public:
  Generator(StringRef dialectName) : dialectName(dialectName) {}

  // Returns whether successfully created output dirs/files.
  bool init(const std::filesystem::path &mainFileRoot);

  // Returns whether successfully terminated output files.
  bool fin(raw_ostream &os);

  // Returns whether successfully emitted attribute/type parsers.
  bool emitParse(StringRef kind, Record &x);

  // Returns whether successfully emitted attribute/type printers.
  bool emitPrint(StringRef kind, StringRef type, ArrayRef<Record *> vec);

  // Emits parse dispatch table.
  bool emitDispatch(StringRef kind, ArrayRef<Record *> vec);

private:
  // Emits header for parse method.
  bool emitParseHelper(StringRef kind, StringRef returnType, StringRef builder,
                       ArrayRef<Init *> args, ArrayRef<std::string> argNames,
                       mlir::raw_indented_ostream &ios);

  StringRef dialectName;
  std::string topStr, bottomStr;
};

bool Generator::init(const std::filesystem::path &mainFileRoot) {
  {
    // Generate top section.
    raw_string_ostream os(topStr);
    os << formatv(licenseHeader, dialectName);

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
        llvm::errs() << "FAILURE: " << incFileOr.getError().message();
        return true;
      }
    }
  }

  {
    // Generate bottom section.
    raw_string_ostream os(topStr);
    // Inject additional cpp include file.
    auto incCppFileName =
        mainFileRoot / formatv("{0}Inc.cpp", dialectName).str();
    if (std::filesystem::exists(incCppFileName)) {
      auto incFileOr =
          MemoryBuffer::getFile(incCppFileName.string(), /*IsText=*/true,
                                /*RequiresNullTerminator=*/false);
      if (incFileOr) {
        os << incFileOr->get()->getBuffer() << "\n";
      } else {
        llvm::errs() << "FAILURE: " << incFileOr.getError().message();
        return true;
      }
    }
  }

  return false;
}

bool Generator::fin(raw_ostream &os) {
  os << topStr << "\n" << StringRef(bottomStr).rtrim() << "\n";
  return false;
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
  if (cType.empty())
    return formatv(format.c_str(), def->getName().str());
  return formatv(format.c_str(), cType.str());
}

bool Generator::emitDispatch(StringRef kind, ArrayRef<Record *> vec) {
  raw_string_ostream sos(bottomStr);
  mlir::raw_indented_ostream os(sos);
  char const *head = R"({1} {0}DialectBytecodeInterface::read{1}(
      DialectBytecodeReader &reader)";
  os << formatv(head, dialectName, capitalize(kind));
  auto funScope = os.scope(" {\n", "}\n\n");

  os << "uint64_t kind;\n";
  os << "if (failed(dialectReader.readVarInt(kind)))\n"
     << "  return " << capitalize(kind) << "();\n";
  os << "switch (kind) ";
  {
    auto switchScope = os.scope("{\n", "}\n");
    for (auto it : vec) {
      os << formatv("case /* {0} */ {1}:\n  return read{0}(dialectReader);\n",
                    it->getName(), it->getValueAsInt("enum"));
    }
    os << "default:\n"
       << "  reader.emitError() << \"unknown builtin attribute code: \" "
       << "<< kind;\n"
       << "  return " << capitalize(kind) << "();\n";
  }
  os << "\nreturn mlirBytecodeSuccess();\n";

  return false;
}

bool Generator::emitParse(StringRef kind, Record &attr) {
  char const *head =
      R"(static {0} read{1}(MlirContext* context, DialectBytecodeReader &reader) )";
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
  return emitParseHelper(kind, returnType, builder, members->getArgs(),
                         argNames, ios);
}

bool Generator::emitParseHelper(StringRef kind, StringRef returnType,
                                StringRef builder, ArrayRef<Init *> args,
                                ArrayRef<std::string> argNames,
                                mlir::raw_indented_ostream &ios) {
  auto funScope = ios.scope("{\n", "}\n\n");

  // Parses for trivial constructors handled in dispatch.
  if (args.empty()) {
    ios << formatv("return {0}::get(context);\n", returnType);
    return false;
  }

  // Print decls.
  StringRef lastCType = "";
  for (auto it : zip(args, argNames)) {
    DefInit *first = dyn_cast<DefInit>(std::get<0>(it));
    if (!first)
      return reportError("Unexpected type for " + std::get<1>(it));
    Record *def = first->getDef();

    std::string cType = getCType(def);
    if (lastCType == cType) {
      ios << ", ";
    } else {
      if (!lastCType.empty())
        ios << ";\n";
      ios << cType << " ";
    }
    ios << std::get<1>(it);
    lastCType = cType;
  }
  ios << ";\n";

  auto listHelperName = [](StringRef name) {
    return formatv("read{0}", capitalize(name));
  };

  // Emit list helper functions.
  for (auto it : zip(args, argNames)) {
    Record *attr = cast<DefInit>(std::get<0>(it))->getDef();
    if (!attr->isSubClassOf("Array"))
      continue;

    // TODO: Dedupe readers.
    Record *def = attr->getValueAsDef("elemT");
    std::string returnType = getCType(def);
    llvm::errs() << std::get<1>(it) << " <<\n";
    ios << "auto " << listHelperName(std::get<1>(it))
        << " = [&]() -> FailureOr<" << returnType << "> ";
    SmallVector<Init *> args;
    SmallVector<std::string> argNames;
    // If a composite bytecode.
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
    StringRef builder = attr->getValueAsString("cBuilder");
    if (emitParseHelper(kind, returnType, builder, args, argNames, ios))
      return true;
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
          if (attr->isSubClassOf("Array")) {
            parser = ("succeeded({0}.readList({2}, " +
                      listHelperName(std::get<1>(it)) + "))")
                         .str();

          } else {
            parser = attr->getValueAsString("cParser").str();
          }
          std::string type = getCType(attr);
          ios << formatv(parser.c_str(), "reader", type, std::get<1>(it));
        },
        [&]() { ios << " &&\n"; });
  }

  //
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
  ios << "}\nreturn " << returnType << "();\n";

  return false;
}

bool Generator::emitPrint(StringRef kind, StringRef type,
                          ArrayRef<Record *> vec) {
  char const *head =
      R"(static void write({0} {1}, DialectBytecodeWriter &writer) const )";
  raw_string_ostream os(bottomStr);
  mlir::raw_indented_ostream ios(os);
  ios << formatv(head, type, kind);
  auto funScope = ios.scope("{\n", "}\n\n");

  ios << "// " << kind << " " << type << " " << vec.size() << "\n";
  for (Record *rec : vec) {
    StringRef pred = rec->getValueAsString("printerPredicate");
    if (!pred.empty()) {
      ios << "if (" << formatv(pred.str().c_str(), kind) << ") {\n";
      ios.indent();
    }

    ios << "writer.writeVarInt(/* " << rec->getName() << " */ "
        << rec->getValueAsInt("enum") << ");\n";

    auto members = rec->getValueAsDag("members");
    for (auto it : llvm::zip(members->getArgs(), members->getArgNames())) {
      ios << "// " << std::get<1>(it)->getAsUnquotedString() << 
         " " << std::get<0>(it)->getAsUnquotedString()
          << "\n";
      DefInit *def = dyn_cast<DefInit>(std::get<0>(it));
      assert(def);
      Record *memberRec = def->getDef();
      // TODO
      if (memberRec->isSubClassOf("Array"))
        continue;

      ios << formatv(memberRec->getValueAsString("cPrinter").str().c_str(),
                     "writer", kind, "FOO")
          << ";\n";
    }

    if (!pred.empty()) {
      ios.unindent();
      ios << "}\n";
    }
  }

  return false;
}

struct AttrOrType {
  std::vector<Record *> attr, type;
};

static bool tableGenMain(raw_ostream &os, RecordKeeper &records) {
  MapVector<StringRef, AttrOrType> dialectAttrOrType;
  Record *attr = records.getClass("DialectAttribute");
  Record *type = records.getClass("DialectType");
  for (auto &it : records.getAllDerivedDefinitions("DialectAttrOrType")) {
    if (it->isSubClassOf(attr)) {
      dialectAttrOrType[it->getValueAsString("dialect")].attr.push_back(it);
    } else if (it->isSubClassOf(type)) {
      dialectAttrOrType[it->getValueAsString("dialect")].type.push_back(it);
    }
  }

  if (dialectAttrOrType.size() != 1)
    return reportError("Only single dialect per invocation allowed");

  // Compare two records by enum value.
  auto compEnum = [](Record *lhs, Record *rhs) -> int {
    return lhs->getValueAsInt("enum") < rhs->getValueAsInt("enum");
  };

  auto compCType = [](Record *lhs, Record *rhs) -> int {
    return lhs->getValueAsInt("cType") < rhs->getValueAsInt("cType");
  };

  auto mainFile =
      std::filesystem::path(records.getInputFilename()).remove_filename();

  auto it = dialectAttrOrType.front();
  Generator gen(it.first);
  if (gen.init(mainFile))
    return true;

  {
    // Handle Attribute emission.
    std::vector<Record *> &vec = it.second.attr;
    std::sort(vec.begin(), vec.end(), compEnum);
    std::map<std::string, std::vector<Record *>> perType;
    for (auto kt : vec)
      perType[getCType(kt)].push_back(kt);
    for (auto jt : perType) {
      llvm::errs() << jt.first << " <<\n";
      for (auto kt : jt.second)
        if (gen.emitParse("attribute", *kt))
          return true;
      if (gen.emitPrint("attribute", jt.first, jt.second))
        return true;
    }
    gen.emitDispatch("attribute", vec);
  }

  {
    // Handle Type emission.
    std::vector<Record *> &vec = it.second.type;
    std::sort(vec.begin(), vec.end(), compEnum);
    for (auto kt : vec)
      if (gen.emitParse("type", *kt))
        return true;
    gen.emitDispatch("type", vec);
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