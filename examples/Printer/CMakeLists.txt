# Instance of using parser
add_executable(MlirBytecodePrintParse PrintParse.c ${MLIR_BYTECODE_PARSE_SOURCES})
if(NOT MSVC)
    target_link_libraries(MlirBytecodePrintParse PUBLIC m)
endif()