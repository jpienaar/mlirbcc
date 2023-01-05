//===- Builder.cpp - Testing bytecode reader ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "llvm/Support/SourceMgr.h"

// Include bytecode parsing implementation.
#include "mlirbcc/Parse.c.inc"

MlirBytecodeStatus
mlirBytecodeOperationStatePush(void *callerState, MlirBytecodeOpHandle name,
                               MlirBytecodeLocHandle loc,
                               MlirBytecodeOperationStateHandle *opState) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeOperationStatePop(void *callerState,
                              MlirBytecodeOperationStateHandle) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeOperationStateAddAttributes(void *callerState,
                                        MlirBytecodeOperationStateHandle,
                                        MlirBytecodeAttrHandle) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddResultTypes(
    void *callerState, MlirBytecodeOperationStateHandle, MlirBytecodeStream *,
    uint64_t numResults) {
  return mlirBytecodeUnhandled();
}

// Note: This _requires_ that the stream is progressed post the last item before
// this function returns.
MlirBytecodeStatus mlirBytecodeOperationStateAddOperands(
    void *callerState, MlirBytecodeOperationStateHandle, MlirBytecodeStream *,
    uint64_t numOperands) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddRegions(
    void *callerState, MlirBytecodeOperationStateHandle, uint64_t numRegions) {
  return mlirBytecodeUnhandled();
}

// Note: This _requires_ that the stream is progressed post the last item before
// this function returns.
MlirBytecodeStatus mlirBytecodeOperationStateAddSuccessors(
    void *callerState, MlirBytecodeOperationStateHandle, MlirBytecodeStream *,
    uint64_t numSuccessors) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeRegionPush(void *callerState,
                                          MlirBytecodeOperationStateHandle,
                                          size_t numBlocks, size_t numValues) {
  return mlirBytecodeUnhandled();
}

// Note: This _requires_ that the stream is progressed post the last item before
// this function returns.
MlirBytecodeStatus
mlirBytecodeOperationStateBlockPush(void *callerState,
                                    MlirBytecodeOperationStateHandle,
                                    MlirBytecodeStream *, uint64_t numArgs) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeOperationStateBlockPop(void *callerState,
                                   MlirBytecodeOperationStateHandle) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeOperationStateRegionPop(void *callerState,
                                    MlirBytecodeOperationStateHandle) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeProcessAttribute(void *callerState,
                                                MlirBytecodeAttrHandle handle) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeProcessType(void *callerState,
                                           MlirBytecodeTypeHandle handle) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeAttrCallBack(void *, MlirBytecodeAttrHandle,
                                            size_t total,
                                            MlirBytecodeDialectHandle,
                                            MlirBytecodeBytesRef,
                                            bool hasCustom) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeTypeCallBack(void *, MlirBytecodeTypeHandle,
                                            size_t total,
                                            MlirBytecodeDialectHandle,
                                            MlirBytecodeBytesRef,
                                            bool hasCustom) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeDialectCallBack(void *,
                                               MlirBytecodeDialectHandle,
                                               size_t total,
                                               MlirBytecodeStringHandle) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeDialectOpCallBack(void *,
                                                 MlirBytecodeDialectHandle,
                                                 MlirBytecodeOpHandle,
                                                 MlirBytecodeStringHandle) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus mlirBytecodeResourceCallBack(
    void *, MlirBytecodeStringHandle groupKey, MlirBytecodeSize totalGroups,
    MlirBytecodeStringHandle resourceKey, MlirBytecodeAsmResourceEntryKind,
    const MlirBytecodeBytesRef *blob, const uint8_t *MlirBytecodeStatusResource,
    const MlirBytecodeStringHandle *) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeBytesRef
mlirBytecodeGetStringSectionValue(void *callerState,
                                  const MlirBytecodeFile *mlirFile,
                                  MlirBytecodeStringHandle idx) {
  return MlirBytecodeBytesRef{.data = 0, .length = 0};
}

MlirBytecodeStatus mlirBytecodeStringCallBack(void *, MlirBytecodeStringHandle,
                                              size_t total,
                                              MlirBytecodeBytesRef) {
  return mlirBytecodeUnhandled();
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s file.mlirbc\n", argv[0]);
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(argv[1]);
  if (std::error_code error = fileOrErr.getError()) {
    return mlirBytecodeEmitError("MlirBytecodeFailed to open file '%s'",
                                 argv[1]),
           1;
  }
  auto idx = sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  const llvm::MemoryBuffer *buffer = sourceMgr.getMemoryBuffer(idx);

  MlirBytecodeBytesRef ref = {.data = (const uint8_t *)buffer->getBufferStart(),
                              .length = buffer->getBufferSize()};
  MlirBytecodeFile mlirFile = mlirBytecodePopulateFile(ref);

  if (!mlirBytecodeFileEmpty(&mlirFile) &&
      mlirBytecodeFailed(mlirBytecodeParseFile(nullptr, ref)))
    return mlirBytecodeEmitError("MlirBytecodeFailed to parse file"), 1;

  return 0;
}
