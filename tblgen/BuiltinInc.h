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

static LogicalResult readAPIntWithKnownWidth(DialectBytecodeReader &reader,
                                             Type type, APInt &val) {
  unsigned bitWidth = getIntegerBitWidth(reader, type);
  if (bitWidth == 0)
    return failure();
  FailureOr<APInt> value = reader.readAPIntWithKnownWidth(bitWidth);
  if (failed(value))
    return failure();
  val = *value;
  return success();
}

static LogicalResult
readAPFloatWithKnownSemantics(DialectBytecodeReader &reader, Type type,
                              APFloat &val) {
  // TODO
  return failure();
}

template <typename T, typename... Ts>
static LogicalResult readResourceHandle(DialectBytecodeReader &reader, T &value,
                                        Ts &&...params) {
  FailureOr<T> handle = reader.readResourceHandle<T>();
  if (failed(handle))
    return failure();
  if (auto *result = dyn_cast<T>(&*handle)) {
    value = std::move(*result);
    return success();
  }
  return failure();
}

template <typename T, typename... Ts>
using has_get_method = decltype(T::get(std::declval<Ts>()...));

template <typename T, typename... Ts>
auto get(MLIRContext *context, Ts &&...params) {
  // Prefer a direct `get` method if one exists.
  if constexpr (llvm::is_detected<has_get_method, T, Ts...>::value) {
    (void)context;
    return T::get(std::forward<Ts>(params)...);
  } else if constexpr (llvm::is_detected<has_get_method, T, MLIRContext *,
                                         Ts...>::value) {
    return T::get(context, std::forward<Ts>(params)...);
  } else {
    // Otherwise, pass to the base get.
    return T::Base::get(context, std::forward<Ts>(params)...);
  }
}