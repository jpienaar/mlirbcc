# mlirbcc C Reader Architecture

`mlirbcc` is designed as a lightweight, dependency-free C implementation of the MLIR Bytecode reader.

## Event-Driven Design
It follows a SAX-like model where the parser triggers callbacks (events) for each element encountered in the bytecode stream:
- `mlirBytecodeIRSectionEnter`
- `mlirBytecodeOperationStatePush` / `Pop`
- `mlirBytecodeOperationRegionPush` / `Pop`
- `mlirBytecodeOperationBlockPush` / `Pop`
- `mlirBytecodeDialectVersionCallBack` (v1+)
- `mlirBytecodeDialectOpWithRegisteredCallBack` (v5+)
- `mlirBytecodeOperationStateAddProperties` (v5+)

## Stack-Based IR Parsing
Instead of a recursive descent parser, `mlirbcc` uses an explicit stack to manage the state of the IR walk. This prevents stack overflow on deep IR structures.

### Stack Structure
The `mbci_MlirIRSectionStack` tracks the remaining counts for operations, regions, and blocks at each nesting level.

```c
struct mbci_MlirIRSectionStackEntry {
  union {
    MlirBytecodeOperationStateHandle opState;
    MlirBytecodeOperationHandle op;
  };
  uint16_t numRegionsRemaining : 15;
  uint16_t isIsolatedFromAbove : 1;
  uint16_t numBlocksRemaining;
  uint32_t numOpsRemaining;
  MlirBytecodeStream *isolatedStream;
};
```

### Parsing Loop
The `mlirBytecodeParseIRSection` function contains a single `while` loop that inspects the top of the stack and decides whether to:
1.  **Pop** a block/region/op if its children are fully parsed.
2.  **Parse a Region** header if `numBlocksRemaining` is unset.
3.  **Parse a Block** header if `numOpsRemaining` is unset.
4.  **Parse an Operation** if `numOpsRemaining > 0`.

#### Active Stream Handling (v2+)
To support isolated regions stored in separate sub-sections (Lazy Loading), the loop maintains an `activeStream` pointer. 
- **Stream Selection**: The parser identifies the `activeStream` by walking **up** the stack from the current level to the most recent entry with a non-null `isolatedStream`. If none is found, it defaults to the primary IR section stream.
- This ensures that operations nested within an isolated region correctly continue to use that region's data stream, even if they themselves are not isolated.
- **Sub-stream Lifetime**: To avoid dynamic allocation, `mlirbcc` uses a static pool of `MlirBytecodeStream` objects indexed by stack depth.
- **Stack Initialization**: `mbci_mlirIRSectionStackPush` resets the `isolatedStream` to `NULL` for each level to ensure clean state transitions.
- **Stale State Prevention**: Because stack entries are reused, bitfields like `isIsolatedFromAbove` must be explicitly initialized or reset for every operation parsed to prevent values from previous operations leaking into subsequent ones at the same depth.
- **Detection Logic**: The `isIsolatedFromAbove` bit is faithfully preserved across the stack push to inform `mbci_parseRegion` whether to expect an inline `kIR` section header.

## Component Sections
`mlirbcc` supports the following top-level sections:
- `mbci_kString` (0): String pool.
- `mbci_kDialect` (1): Dialect names and op names.
- `mbci_kAttrType` (2): Attributes and Types.
- `mbci_kAttrTypeOffset` (3): Offsets into AttrType section.
- `mbci_kIR` (4): The actual IR content.
- `mbci_kResource` (5): External data resources.
- `mbci_kResourceOffset` (6): Offsets into Resource section.
- `mbci_kDialectVersions` (7): Versioning information for dialects (v1+).
- `mbci_kProperties` (8): Native operation properties (v5+).

## Bytecode Version Support
The reader has been updated to support bytecode versions up to **v6**.

### Updates beyond v0:
- **Version Validation**: Validates `version <= 6`.
- **Flexible Section Presence**: `mbci_isSectionOptional()` is now version-aware (e.g., properties are optional if version < 5).
- **New Sections**: Integration of Dialect Versioning (7) and Properties (8).
- **Variable-Width Flags**: Correctly handles `(value << 1) | flag` encoding for dialect names (v1+), operation names (v5+), and block arguments (v4+).
- **Lazy Loading (v2+)**: Full support for isolated regions stored as inline IR sub-sections, including:
  - **Deferred Region Storage**: `mlirBytecodeStoreDeferredRegionData` stores raw byte ranges for later materialization.
  - **Multi-Region Support**: `mlirBytecodeGetOperationNumRegions` queries actual region count from operations.
  - **Materialization**: `mlirBytecodeParseDeferredRegion` parses stored data with proper region/block handling.
  - **Version Gating**: Lazy loading only enabled for `bytecodeVersion >= 2`.

- Extension of `uselist` ordering logic to allow sorting if required by the consumer.

## 8. LLVM Adapter Architecture

The integration of the C bytecode kernel into LLVM's `BytecodeReader.cpp` uses an **Adapter Pattern** to bridge the procedural C API with the object-oriented MLIR C++ API.

### 8.1 Entry Point: `readBytecodeFileImpl`
The primary entry point in `BytecodeReader.cpp` is `readBytecodeFileImpl`. It performs the following orchestration:
1.  **State Initialization**: Creates a `ParsingState` **before** calling `mlirBytecodePopulateParserState` to ensure valid error context for version validation and other early errors.
2.  **Version Check**: Returns failure if parser state is empty (version too new, malformed file, etc.).
3.  **Section Pre-parsing**: Manually extracts the **Properties Section** (v5+) to build an offset table before the main IR walk begins.
4.  **Kernel Invocation**: Calls `mlirBytecodeParse` (from `Parse.c.inc`), which executes the stack-based IR walk.
5.  **Post-Parsing Passes**:
    *   **Lazy Loading Registration**: Links lazy-loadable operations to the reader's implementation.
    *   **Use-List Reordering**: (Planned) Sorter pass to apply captured permutation indices to IR Values.

### 8.2 Callback Bridge
The C++ adapter implements the `MLIRBC_DEF` callbacks. Key bridges include:
-   **Value Mapping**: The adapter maintains a `ValueScope` stack that maps the parser's value IDs into actual `mlir::Value` objects.
-   **Operation Construction**: `mlirBytecodeOperationStatePush` initializes an `mlir::OperationState`, and `mlirBytecodeOperationStatePop` calls `Operation::create(opState)` to materialize the operation. Crucially, the resulting `Operation*` is stored in the `id` field of the `MlirBytecodeOperationHandle`, serving as an opaque pointer for subsequent callbacks (like use-list ordering or property application).
-   **Properties Resolution**: For v5+ operations, the `mlirBytecodeOperationStateAddProperties` callback uses the pre-built offset table to locate the property blob and calls the `BytecodeOpInterface::readProperties` method on the materialized op.

### 8.3 Use-List Delivery Mechanism
The C kernel delivering indices to the adapter was implemented by adding new callbacks:
- `mlirBytecodeBlockArgAddUseListOrder`
- `mlirBytecodeResultAddUseListOrder`

These callbacks deliver the permutation indices parsed from the v3+ bytecode. The kernel uses a **two-tier buffer management strategy**: a 64-element stack buffer for common cases and automatic heap allocation (`malloc`) for larger lists, ensuring efficient delivery to the C++ adapter's `valueToUseListMap`.

### 8.4 Error Handling
Robust error handling ensures graceful failures without crashes:
- **Forward Reference Cleanup**: When unresolved forward references are detected, `op.dropAllUses()` is called on all placeholder ops before returning failure. This prevents "operation destroyed but still has uses" crashes.
- **Trailing Character Detection**: `parseAttribute` is called with `&numRead` to detect unconsumed characters, with proper error emission in bytecode format.
- **Empty Parser State**: If `mlirBytecodePopulateParserState` fails (e.g., version > 6), the adapter returns `mlirBytecodeFailure()` not success.

### 8.5 Lazy Loading Materialization
The `BytecodeReader::Impl::materialize` function implements deferred region parsing:
1. Retrieves stored `irData` from `lazyLoadableOpsMap`
2. Sets up `regionStack` and `valueScopes` for parsing context
3. Calls `mlirBytecodeParseDeferredRegion` with stored parser state
4. Nested isolated operations are automatically tracked for subsequent materialization
