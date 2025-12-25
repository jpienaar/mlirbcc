#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

// Forward declarations as required by Parse.c.inc
typedef struct MlirBytecodeOperationState MlirBytecodeOperationState;
typedef struct MlirBytecodeOperation MlirBytecodeOperation;

struct MlirBytecodeOperationState {
  int dummy;
};

struct MlirBytecodeOperation {
  int dummy;
};

#define MLIRBC_DEF static
#include "mlirbcc/Parse.c.inc"

// Stub implementations of all extension points.

void mlirBytecodeIRSectionEnter(void *context, void *retBlock) {}

MlirBytecodeStatus
mlirBytecodeOperationStatePush(void *context, MlirBytecodeOpHandle name,
                               MlirBytecodeLocHandle loc,
                               MlirBytecodeOperationStateHandle *opState) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddAttributeDictionary(
    void *context, MlirBytecodeOperationStateHandle state,
    MlirBytecodeAttrHandle attr) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationStateAddResultTypes(void *context,
                                         MlirBytecodeOperationStateHandle state,
                                         MlirBytecodeSize numResults) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationStateAddResultType(void *context,
                                        MlirBytecodeOperationStateHandle state,
                                        MlirBytecodeTypeHandle type) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationStateAddOperands(void *context,
                                      MlirBytecodeOperationStateHandle state,
                                      MlirBytecodeSize numOperands) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationStateAddOperand(void *context,
                                     MlirBytecodeOperationStateHandle state,
                                     MlirBytecodeValueHandle value) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddRegions(
    void *context, MlirBytecodeOperationStateHandle state, uint64_t numRegions,
    bool isIsolatedFromAbove) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationStateAddSuccessors(void *context,
                                        MlirBytecodeOperationStateHandle state,
                                        MlirBytecodeSize numSuccessors) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationStateAddSuccessor(void *context,
                                       MlirBytecodeOperationStateHandle state,
                                       MlirBytecodeHandle successor) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationStateAddProperties(void *context,
                                        MlirBytecodeOperationStateHandle state,
                                        MlirBytecodeHandle propertiesIndex) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationStatePop(void *context,
                              MlirBytecodeOperationStateHandle opStateHandle,
                              MlirBytecodeOperationHandle *opHandle) {
  *opHandle = NULL;
  return mlirBytecodeSuccess();
}

bool mlirBytecodeOperationWasLazyLoaded(void *context,
                                        MlirBytecodeOperationHandle opHandle) {
  return false;
}

void mlirBytecodeStoreDeferredRegionData(void *context,
                                         MlirBytecodeOperationHandle opHandle,
                                         const uint8_t *data, uint64_t length) {
}

uint64_t
mlirBytecodeGetOperationNumRegions(MlirBytecodeOperationHandle opHandle) {
  return 0;
}

MlirBytecodeStatus
mlirBytecodeOperationRegionPush(void *context, MlirBytecodeOperationHandle op,
                                size_t numBlocks, size_t numValues) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationBlockPush(void *context, MlirBytecodeOperationHandle op,
                               MlirBytecodeSize numArgs) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationBlockAddArgument(
    void *context, MlirBytecodeOperationHandle op, MlirBytecodeTypeHandle type,
    MlirBytecodeLocHandle loc) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeBlockArgAddUseListOrder(void *context,
                                                       uint64_t valueIndex,
                                                       bool indexPairEncoding,
                                                       const uint64_t *indices,
                                                       uint64_t numIndices) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeResultAddUseListOrder(
    void *context, MlirBytecodeOperationHandle op, uint64_t resultIndex,
    bool indexPairEncoding, const uint64_t *indices, uint64_t numIndices) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationBlockPop(void *context, MlirBytecodeOperationHandle op) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationRegionPop(void *context, MlirBytecodeOperationHandle op) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeAttributesPush(void *context,
                                              MlirBytecodeSize numAttributes) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeAssociateAttributeRange(
    void *context, MlirBytecodeAttrHandle handle,
    MlirBytecodeDialectHandle dialect, MlirBytecodeBytesRef bytes,
    bool hasCustom) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeTypesPush(void *context,
                                         MlirBytecodeSize numTypes) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeAssociateTypeRange(void *context, MlirBytecodeTypeHandle handle,
                               MlirBytecodeDialectHandle dialect,
                               MlirBytecodeBytesRef bytes, bool hasCustom) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeDialectsPush(void *context,
                                            MlirBytecodeSize numDialects) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeDialectCallBack(void *context, MlirBytecodeDialectHandle dialect,
                            MlirBytecodeStringHandle name) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeDialectOpNames(void *context,
                                              MlirBytecodeSize numOps) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeDialectOpCallBack(void *context, MlirBytecodeOpHandle op,
                              MlirBytecodeDialectHandle dialect,
                              MlirBytecodeStringHandle name) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeDialectVersionCallBack(void *context,
                                   MlirBytecodeDialectHandle dialect,
                                   MlirBytecodeBytesRef version) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeDialectOpWithRegisteredCallBack(
    void *context, MlirBytecodeOpHandle op, MlirBytecodeDialectHandle dialect,
    MlirBytecodeStringHandle name, bool isRegistered) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeResourceDialectGroupEnter(void *context,
                                      MlirBytecodeDialectHandle dialect,
                                      MlirBytecodeSize numResources) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeResourceExternalGroupEnter(void *context,
                                       MlirBytecodeStringHandle groupKey,
                                       MlirBytecodeSize numResources) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeResourceBlobCallBack(
    void *context, MlirBytecodeStringHandle resourceKey,
    MlirBytecodeSize alignment, MlirBytecodeBytesRef blob) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeResourceBoolCallBack(
    void *context, MlirBytecodeStringHandle resourceKey, const uint8_t value) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeResourceEmptyCallBack(void *context,
                                  MlirBytecodeStringHandle resourceKey) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeResourceStringCallBack(void *context,
                                   MlirBytecodeStringHandle resourceKey,
                                   MlirBytecodeStringHandle value) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeStringsPush(void *context,
                                           MlirBytecodeSize numStrings) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeAssociateStringRange(void *context, MlirBytecodeStringHandle handle,
                                 MlirBytecodeBytesRef bytes) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeParseAttribute(void *context,
                                              MlirBytecodeAttrHandle handle) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeParseType(void *context,
                                         MlirBytecodeTypeHandle handle) {
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeGetStringSectionValue(void *context, MlirBytecodeStringHandle idx,
                                  MlirBytecodeBytesRef *result) {
  return mlirBytecodeSuccess();
}

void *mlirBytecodeAllocateTemp(void *context, size_t bytes) { return NULL; }

void mlirBytecodeFreeTemp(void *context, void *ptr) {}

#ifdef MLIRBC_STUB_REALISTIC
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc < 2)
    return 1;
  int fd = open(argv[1], O_RDONLY);
  if (fd == -1)
    return 1;
  struct stat fileInfo;
  if (fstat(fd, &fileInfo) == -1) {
    close(fd);
    return 1;
  }
  uint8_t *stream =
      (uint8_t *)mmap(0, fileInfo.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (stream == MAP_FAILED) {
    close(fd);
    return 1;
  }

  MlirBytecodeParserState state;
  MlirBytecodeBytesRef bytes = {stream, (size_t)fileInfo.st_size};
  state = mlirBytecodePopulateParserState(NULL, bytes);
  if (!mlirBytecodeParserStateEmpty(&state)) {
    mlirBytecodeParse(NULL, &state, NULL);
  }

  munmap(stream, fileInfo.st_size);
  close(fd);
  return 0;
}
#else
int main(int argc, char **argv) {
  MlirBytecodeParserState state = {0};
  MlirBytecodeBytesRef bytes = {NULL, 0};
  state = mlirBytecodePopulateParserState(NULL, bytes);
  if (!mlirBytecodeParserStateEmpty(&state)) {
    mlirBytecodeParse(NULL, &state, NULL);
  }
  return 0;
}
#endif
