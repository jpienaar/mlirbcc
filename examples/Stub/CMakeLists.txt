# Stub example for verifying C99 and LTO optimization
add_executable(MlirBytecodeStub StubParse.c)
set_property(TARGET MlirBytecodeStub PROPERTY C_STANDARD 99)
target_include_directories(MlirBytecodeStub PRIVATE ../../include)

if(NOT MSVC)
    target_link_libraries(MlirBytecodeStub PUBLIC m)
endif()
