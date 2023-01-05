add_executable(MlirBytecodeBuilder
  Builder.cpp
  ${MLIR_BYTECODE_PARSE_SOURCES}
)
target_link_libraries(MlirBytecodeBuilder MLIRCAPIIR)