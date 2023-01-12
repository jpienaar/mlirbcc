add_executable(MlirBytecodeBuilder
  Builder.cpp
)
target_link_libraries(MlirBytecodeBuilder MLIRAsmParser MLIRIR LLVMSupport MLIRSupport)
