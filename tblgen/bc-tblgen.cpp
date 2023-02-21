
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

  // Returns whether successfully emitted attribute defs.
  bool emitParse(StringRef kind, Record &x);

  // Emits header for parse method.
  bool emitParseHeader(StringRef kind, Record &x);

  // Emits dispatch table.
  bool emitDispatch(StringRef kind, ArrayRef<Record *> vec);

private:
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

    os << formatv(R"(
// Entry point for {0} dialect Attribute parsing.
MlirBytecodeStatus
mlirBytecodeParse{0}Attr(void *state, MlirBytecodeAttrHandle attrHandle,
                             MlirBytecodeBytesRef str, bool hasCustom);

// Entry point for {0} dialect Type parsing.
MlirBytecodeStatus
mlirBytecodeParse{0}Type(void *state, MlirBytecodeTypeHandle typeHandle,
                             MlirBytecodeBytesRef str, bool hasCustom);

)",
                  dialectName);
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

bool Generator::emitParseHeader(StringRef kind, Record &attr) {
  char const *head =
      R"(static {0} read{1}(MlirContext* context, DialectBytecodeReader &reader) )";
  raw_string_ostream os(bottomStr);
  mlir::raw_indented_ostream ios(os);
  std::string returnType = getCType(&attr);
  ios << formatv(head, returnType, attr.getName());
  return false;
}

bool Generator::emitParse(StringRef kind, Record &attr) {
  DagInit *members = attr.getValueAsDag("members");
  std::string returnType = getCType(&attr);
  raw_string_ostream os(bottomStr);
  mlir::raw_indented_ostream ios(os);
  auto funScope = ios.scope("{\n", "}\n\n");

  // Parses for trivial constructors handled in dispatch.
  if (!members->getNumArgs()) {
    ios << formatv("return {0}::get(context);\n", getCType(&attr));
    return false;
  }

  // Print decls.
  StringRef lastCType = "";
  for (auto it : zip(members->getArgs(), members->getArgNames())) {
    DefInit *first = dyn_cast<DefInit>(std::get<0>(it));
    if (!first) {
      return reportError("Unexpected type for " +
                         std::get<1>(it)->getAsString());
    }
    Record *def = first->getDef();

    std::string cType = getCType(def);
    if (lastCType == cType) {
      ios << ", ";
    } else {
      if (!lastCType.empty())
        ios << ";\n";
      ios << cType << " ";
    }
    ios << std::get<1>(it)->getAsUnquotedString();
    lastCType = cType;
  }
  ios << ";\n";

  auto listHelperName = [](StringInit *name) {
    return formatv("read{0}", capitalize(name->getAsUnquotedString()));
  };

  // Emit list helper functions.
  for (auto it : zip(members->getArgs(), members->getArgNames())) {
    Record *attr = cast<DefInit>(std::get<0>(it))->getDef();
    if (!attr->isSubClassOf("Array"))
      continue;
    // TODO: Should be recursive call.
    ios << "// TODO: " << listHelperName(std::get<1>(it)) << "\n";
  }

  // Print parse conditional.
  {
    ios << "if ";
    auto parenScope = ios.scope("(", ") {");
    ios.indent();

    auto parsedArgs = llvm::to_vector(
        make_filter_range(members->getArgs(), [](Init *const attr) {
          Record *def = cast<DefInit>(attr)->getDef();
          if (def->isSubClassOf("Array")) {
            def = def->getValueAsDef("elemT");
          }
          return !def->getValueAsString("cParser").empty();
        }));

    interleave(
        zip(parsedArgs, members->getArgNames()),
        [&](auto it) {
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
          ios << formatv(parser.c_str(), "reader", type,
                         std::get<1>(it)->getAsUnquotedString())
              << ")";
        },
        [&]() { ios << " &&\n"; });
  }

  StringRef postProcess = attr.getValueAsString("postProcess").rtrim();
  if (!postProcess.empty()) {
    ios << "\n";
    ios.printReindented(postProcess);
  }

  // Return the invoked constructor.
  ios << formatv("\nreturn {0}::get(", returnType);

  auto passedArgs = llvm::to_vector(
      make_filter_range(members->getArgNames(), [](StringInit *const str) {
        return !StringRef(str->getAsUnquotedString()).starts_with("_");
      }));
  ios.unindent();
  interleaveComma(passedArgs, ios,
                  [&](StringInit *str) { ios << str->getAsUnquotedString(); });
  ios << ");\n";

  // TODO: Emit error in debug.
  // ios << "}\nreturn mlirBytecodeEmitError(\"invalid " << attr.getName()
  //     << "\");\n";
  ios << "}\nreturn " << returnType << "();\n";

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

  auto mainFile =
      std::filesystem::path(records.getInputFilename()).remove_filename();

  auto it = dialectAttrOrType.front();
  Generator gen(it.first);
  if (gen.init(mainFile

               ))
    return true;

  {
    // Handle Attribute emission.
    std::vector<Record *> &vec = it.second.attr;
    std::sort(vec.begin(), vec.end(), compEnum);
    for (auto kt : vec)
      if (gen.emitParseHeader("attribute", *kt) ||
          gen.emitParse("attribute", *kt))
        return true;
    gen.emitDispatch("attribute", vec);
  }

  {
    // Handle Type emission.
    std::vector<Record *> &vec = it.second.type;
    std::sort(vec.begin(), vec.end(), compEnum);
    for (auto kt : vec)
      if (gen.emitParseHeader("type", *kt) || gen.emitParse("type", *kt))
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