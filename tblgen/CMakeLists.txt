message(DEBUG "Building TableGen helpers")

set(LLVM_OPTIONAL_SOURCES
  BuiltinInc.cpp
)

add_executable(bc-tblgen
  bc-tblgen.cpp
  )
set_target_properties(bc-tblgen PROPERTIES FOLDER "Tablegenning")
target_link_libraries(bc-tblgen
  PRIVATE
  LLVMDemangle
  LLVMSupport
  LLVMTableGen
  MLIRSupportIndentedOstream
  )
  llvm_update_compile_flags(bc-tblgen)