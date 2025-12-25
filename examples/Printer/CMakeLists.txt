# Instance of using parser
add_executable(MlirBytecodePrintParse PrintParse.c ${MLIR_BYTECODE_PARSE_SOURCES})
set_property(TARGET MlirBytecodePrintParse PROPERTY C_STANDARD 99)
if(NOT MSVC)
    target_link_libraries(MlirBytecodePrintParse PUBLIC m)
endif()