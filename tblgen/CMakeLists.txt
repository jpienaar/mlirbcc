message(DEBUG "Building TableGen helpers")

set(LLVM_OPTIONAL_SOURCES
  BuiltinInc.cpp
)

add_executable(mlir-bc-tblgen
  mlir-bc-tblgen.cpp
  )
set_target_properties(mlir-bc-tblgen PROPERTIES FOLDER "Tablegenning")
target_link_libraries(mlir-bc-tblgen
  PRIVATE
  LLVMDemangle
  LLVMSupport
  LLVMTableGen
  MLIRSupportIndentedOstream
  )
  llvm_update_compile_flags(mlir-bc-tblgen)
