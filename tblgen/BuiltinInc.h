#include <stdbool.h>

static LogicalResult readAPIntWithKnownWidth(DialectBytecodeReader &reader,
    unsigned bitWidth, APInt &val) {
  FailureOr<APInt> value = reader.readAPIntWithKnownWidth(bitWidth);
  if (failed(value)) return failure();
  val = *value;
  return success();
}

// Returns the bitwidth if known, else return 0.
static unsigned getIntegerBitWidth(DialectBytecodeReader &reader, Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth();
  } else if (type.isa<IndexType>()) {
    return IndexType::kInternalStorageBitWidth;
  }
  reader.emitError()
        << "expected integer or index type for IntegerAttr, but got: " << type;
  return 0;
}