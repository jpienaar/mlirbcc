message(DEBUG "Building TableGen helpers")

set(LLVM_OPTIONAL_SOURCES
  BuiltinInc.cpp
)

set(LLVM_LINK_COMPONENTS
  Demangle
  Support
  TableGen
)

add_tablegen(bc-tblgen MLIR
  bc-tblgen.cpp
  )
set_target_properties(bc-tblgen PROPERTIES FOLDER "Tablegenning")
target_link_libraries(bc-tblgen
  PRIVATE
  MLIRSupportIndentedOstream
  )