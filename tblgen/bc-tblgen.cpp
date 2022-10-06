
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
    R"(//===-- {0}Parse{1} - Parser driver for {0} dialect ---*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// {0} dialect Attribute & Type parser driver.
//===----------------------------------------------------------------------===//

// The implementation of the dialect parsing is instantiated if
//   #define MLIRBC_PARSE_IMPLEMENTATION
// or
//   #define MLIRBC_{0}_PARSE_IMPLEMENTATION
// is set before including this file.

)";

// Helper class to generate C++ bytecode parser helpers.
class Generator {
public:
  Generator(StringRef dialectName) : dialectName(dialectName) {}

  // Returns whether successfully created output dirs/files.
  bool init(const std::filesystem::path &mainFileRoot);

  // Returns whether successfully terminated output files.
  bool fin();

  // Returns whether successfully emitted attribute defs.
  bool emitParse(StringRef kind, Record &x);
  bool emitCreate(StringRef kind, Record &x);

  bool emitDispatch(StringRef kind, ArrayRef<Record *> vec);

private:
  StringRef dialectName;
  std::unique_ptr<ToolOutputFile> dialectHeaderFile, dialectCppFile;
};

bool Generator::init(const std::filesystem::path &mainFileRoot) {
  const std::filesystem::path incPrefix("include/mlirbcc/Dialect");
  const std::filesystem::path cppPrefix("lib/Dialect");
  std::vector<std::pair<std::reference_wrapper<std::unique_ptr<ToolOutputFile>>,
                        std::filesystem::path>>
      files = {{dialectHeaderFile, incPrefix / dialectName.str() / "Parse.h"},
               {dialectCppFile, cppPrefix / dialectName.str() / "Parse.cpp"}};

  for (auto it : files) {
    std::error_code ec;
    auto dir = std::get<1>(it);
    dir.remove_filename();
    std::filesystem::create_directories(dir, ec);
    if (ec) {
      return reportError("failed to create directory '" + dir.string() +
                         "' as " + ec.message());
    }

    std::string outputFilename = std::get<1>(it).string();
    std::get<0>(it).get() =
        std::make_unique<ToolOutputFile>(outputFilename, ec, sys::fs::OF_Text);
    if (ec) {
      return reportError("error opening " + outputFilename + ": " +
                         ec.message() + "\n");
    }
    std::get<0>(it).get()->keep();
    std::get<0>(it).get()->os()
        << formatv(licenseHeader, dialectName, std::get<1>(it).extension());
  }

  // Emit header.
  raw_ostream &os = dialectHeaderFile->os();
  os << formatv(R"(
#ifndef {0}_PARSE
#define {0}_PARSE

#include "mlirbcc/Parse.h"

#ifdef __cplusplus
extern "C" {
#endif
)",
                dialectName.upper(), dialectName);

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
// Entry point for {1} dialect Attribute parsing.
MlirBytecodeStatus
mlirBytecodeParse{1}Attr(void *state, MlirBytecodeDialectHandle dialectHandle,
  MlirBytecodeAttrHandle attrHandle, size_t total, bool hasCustom,
  MlirBytecodeBytesRef str);

// Entry point for Builtin dialect Type parsing.
MlirBytecodeStatus
mlirBytecodeParse{1}Type(void *state, MlirBytecodeDialectHandle dialectHandle,
    MlirBytecodeTypeHandle typeHandle, size_t total, bool hasCustom,
    MlirBytecodeBytesRef str);

)",
                dialectName.upper(), dialectName);

  dialectCppFile->os() << formatv("#include \"mlirbcc/Dialect/{0}/Parse.h\"\n\n",
                                  dialectName);
  // Inject additional cpp include file.
  auto incCppFileName = mainFileRoot / formatv("{0}Inc.cpp", dialectName).str();
  if (std::filesystem::exists(incCppFileName)) {
    auto incFileOr =
        MemoryBuffer::getFile(incCppFileName.string(), /*IsText=*/true,
                              /*RequiresNullTerminator=*/false);
    if (incFileOr) {
      dialectCppFile->os() << incFileOr->get()->getBuffer() << "\n";
    } else {
      llvm::errs() << "FAILURE: " << incFileOr.getError().message();
      return true;
    }
  }

  return false;
}

bool Generator::fin() {
  raw_ostream &os = dialectHeaderFile->os();
  os << formatv(R"(#ifdef __cplusplus
}
#endif
#endif // {0}_PARSE 
  )",
                dialectName.upper(), dialectName);

  return false;
}

static std::string capitalize(StringRef str) {
  return ((Twine)toUpper(str[0]) + str.drop_front()).str();
}

bool Generator::emitDispatch(StringRef kind, ArrayRef<Record *> vec) {
  mlir::raw_indented_ostream os(dialectCppFile->os());
  char const *head = R"(MlirBytecodeStatus
mlirBytecodeParse{0}{1}(void *state, MlirBytecodeDialectHandle dialectHandle,
    MlirBytecode{1}Handle {2}Handle, size_t total, bool hasCustom,
    MlirBytecodeBytesRef str) )";
  os << formatv(head, dialectName, capitalize(kind), kind);
  auto funScope = os.scope("{\n", "}\n\n");

  os << "if (!hasCustom)\n  return mlirBytecodeUnhandled();\n";
  os << "MlirBytecodeStream stream = mlirBytecodeStreamCreate(str);\n";
  os << "MlirBytecodeStream *pp = &stream;\n";
  os << "MlirBytecodeHandle kind;\n";
  os << "if (mlirBytecodeFailed(mlirBytecodeGetNextHandle(pp, "
        "&kind)))\n  return mlirBytecodeFailure();\n";
  os << "switch (kind) ";
  {
    auto switchScope = os.scope("{\n", "}\n");
    for (auto it : vec) {
      os << formatv(
          "case /* {0} */ {1}:\n  return parse{0}(state, {2}Handle, pp);\n",
          it->getName(), it->getValueAsInt("enum"), kind);
    }
    os << "default:\n  return mlirBytecodeUnhandled();\n";
  }
  os << "\nreturn mlirBytecodeSuccess();\n";

  return false;
}

bool Generator::emitParse(StringRef kind, Record &attr) {
  raw_ostream &os = dialectCppFile->os();
  char const *head = R"(static MlirBytecodeStatus parse{0}(void *bcUserState,
    MlirBytecode{1}Handle bc{1}Handle, MlirBytecodeStream *bcStream) )";
  mlir::raw_indented_ostream ios(os);
  ios << formatv(head, attr.getName(), capitalize(kind));
  auto funScope = ios.scope("{\n", "}\n\n");
  DagInit *members = attr.getValueAsDag("members");

  // Print decls.
  StringRef lastCType = "";
  for (auto it : zip(members->getArgs(), members->getArgNames())) {
    DefInit *first = dyn_cast<DefInit>(std::get<0>(it));
    if (!first) {
      return reportError("Unexpected type for " +
                         std::get<1>(it)->getAsString());
    }
    Record *def = first->getDef();
    StringRef cType = def->getValueAsString("cType");
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

  if (members->getNumArgs() > 0) {
    ios << ";\n";

    ios << "if ";
    auto parenScope = ios.scope("(", ") {");
    ios.indent();

    auto parsedArgs = llvm::to_vector(
        make_filter_range(members->getArgs(), [](Init *const attr) {
          return !cast<DefInit>(attr)
                      ->getDef()
                      ->getValueAsString("cParser")
                      .empty();
        }));

    interleave(
        zip(parsedArgs, members->getArgNames()),
        [&](auto it) {
          Record *attr = cast<DefInit>(std::get<0>(it))->getDef();
          StringRef parser = attr->getValueAsString("cParser");
          ios << "mlirBytecodeSucceeded("
              << formatv(parser.str().c_str(), "bcUserState", "bcStream",
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
  ios << formatv("\nreturn mlirBytecodeCreate{0}{1}(bcUserState, bc{2}Handle",
                 attr.getValueAsString("dialect"), attr.getName(),
                 capitalize(kind));

  auto passedArgs = llvm::to_vector(
      make_filter_range(members->getArgNames(), [](StringInit *const str) {
        return !StringRef(str->getAsUnquotedString()).starts_with("_");
      }));
  if (members->getNumArgs() > 0) {
    ios.unindent();
    ios << ", ";
    interleaveComma(passedArgs, ios, [&](StringInit *str) {
      ios << str->getAsUnquotedString();
    });
  }
  ios << ");\n";

  if (members->getNumArgs() > 0) {
    ios << "}\nreturn mlirBytecodeEmitError(\"invalid " << attr.getName()
        << "\");\n";
  }

  return false;
}

bool Generator::emitCreate(StringRef kind, Record &attr) {
  raw_ostream &headerOs = dialectHeaderFile->os();
  raw_ostream &cppOs = dialectCppFile->os();
  DagInit *members = attr.getValueAsDag("members");

  // Emit extern & weak function for create.
  // - Extern is function user needs to implement;
  // - Weak linkage function is to allow partially defining them;
  headerOs << "// Create method for " << attr.getName() << ".\n";
  headerOs << "extern ";

  if (generateWeak)
    cppOs << "__attribute__((weak)) ";

  SmallVector<ToolOutputFile *> files{dialectHeaderFile.get()};
  if (generateWeak)
    files.push_back(dialectCppFile.get());

  for (auto *file : files) {
    raw_ostream &os = file->os();
    os << "MlirBytecodeStatus mlirBytecodeCreate"
       << attr.getValueAsString("dialect") << attr.getName()
       << formatv("(void *bcUserState, MlirBytecode{0}Handle bc{0}Handle",
                  capitalize(kind));

    if (members->getNumArgs() > 0)
      os << ", ";
    interleaveComma(zip(members->getArgs(), members->getArgNames()), os,
                    [&](auto it) {
                      Record *attr = cast<DefInit>(std::get<0>(it))->getDef();
                      os << attr->getValueAsString("cType") << " "
                         << std::get<1>(it)->getAsUnquotedString();
                    });
  }
  headerOs << ");\n\n";

  if (generateWeak)
    cppOs << ") {\n . return mlirBytecodeUnhandled();\n}\n\n";

  return false;
}

struct AttrOrType {
  std::vector<Record *> attr, type;
};

static bool tableGenMain(raw_ostream &os, RecordKeeper &records) {
  MapVector<StringRef, AttrOrType> dialectAttrOrType;
  Record *attr = records.getClass("DialectAttribute");
  Record *type = records.getClass("DialectType");
  for (auto &it : records.getAllDerivedDefinitions("DialectAttributeOrType")) {
    if (it->isSubClassOf(attr)) {
      dialectAttrOrType[it->getValueAsString("dialect")].attr.push_back(it);
    } else if (it->isSubClassOf(type)) {
      dialectAttrOrType[it->getValueAsString("dialect")].type.push_back(it);
    }
  }

  // Compare two records by enum value.
  auto compEnum = [](Record *lhs, Record *rhs) -> int {
    return lhs->getValueAsInt("enum") < rhs->getValueAsInt("enum");
  };

  auto mainFileRoot=std::filesystem::path(records.getInputFilename())
                .remove_filename();
  for (auto it : dialectAttrOrType) {
    Generator gen(it.first);
    if (gen.init(mainFileRoot))
      return true;

    {
      std::vector<Record *> &vec = it.second.attr;
      std::sort(vec.begin(), vec.end(), compEnum);
      for (auto kt : vec)
        if (gen.emitParse("attr", *kt) || gen.emitCreate("attr", *kt))
          return true;
      gen.emitDispatch("attr", vec);
    }

    {
      std::vector<Record *> &vec = it.second.type;
      std::sort(vec.begin(), vec.end(), compEnum);
      for (auto kt : vec)
        if (gen.emitParse("type", *kt) || gen.emitCreate("type", *kt))
          return true;
      gen.emitDispatch("type", vec);
    }

    gen.fin();
  }

  return false;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(argc, argv);
  progName = argv[0];

  return TableGenMain(argv[0], &tableGenMain);
}