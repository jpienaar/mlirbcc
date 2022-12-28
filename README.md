# MLIR ByteCode (standalone) reader

This is a standalone, pure C, event-driven online algorithm for parsing MLIR
bytecode.

Currently this is incomplete and there is duplication with C++ code, but
goal is to push this to further state where that duplication can be removed.

This project is under active development/being bootstrapped.

### Goals

1. Standalone reader with no dependencies

   Reason: Bytecode format is a spec as well as an implementation. Having
   multiple readers makes this both clearer as well as enables using the
   bytecode in places where one doesn't need/want to depend on MLIR as a whole

2. In-memory representation is configurable/no dedicated in memory format

   Reason: This parser is similar to event-based/SAX style parsing in that we
   have streaming reader that triggers configurable events rather than building
   a single in-memory structure/DOM first. When looking around at different
   different uses we found that the current reader's in-memory structure was
   getting reconverted into a more suitable form and not really used (e.g., it
   had more overhead than utility). Instead allow folks to read and construct
   (if desirable) their own in-memory structure. MLIR already has an in-memory
   structure but it need not be used/is an example of reader implementation.

3. Header C library

   Reason: The structure of the parsing follows a `push`/`pop` structure, where
   `push` creates a variable which `pop` populates. `pop` may be returning a
   pointer to an in-memory block which gets populated rather than a stack
   variable. And coming back to the previous point, the parser shouldn't need to
   know. This configurability and extension is easier/more directly obvious with
   a header library where compiling together with extension points defined are
   obvious.

4. Allow partial parsing/early aborts

   Reason: Applications where one needs more "raw" interactions with the IR
   (e.g., mutate a legacy .mlirbc file that can no longer be read by default
   MLIR reader), or where the consumer of the file has limit capability (e.g.,
   abort if op without registered lowering is in input).

#### Considerations during

1. Externalize all state to caller. The parsers themselves should be stateless
and up to instantiator to manage any caching etc deemed useful for the given
application.

2. Keep allocations inside parser to minimal/let instantiation dictate allocating.

   Note: Some allocations can only be avoided by doing additional work or by
   restricting the spectrum of MLIR files that can be consumed.

### Compilation

Build and testing with:

```sh
cmake -B build && cmake --build build && ./build/examples/Printer/MlirBytecodePrintParse testdata/general.mlirbc
```

Note: building for size constrained one probably wants to set MinSizeRel at
least, enable LTO and disable verbose error reporting and debugging. The optimal
compiler settings have not been explored.

### Structure

- `BytecodeParse.h` is the bytecode parsers;
  Define MLIR_BYTECODE_PARSE_IMPL before including it to instantiate
  implementation;
- `{Dialect}Parse.h` are the `Dialect`'s attribute & type parsers;
  Define MLIR_BYTECODE_PARSE_IMPL or MLIR_BYTECODE_{DIALECT}_PARSE_IMPL before
  including it to instantiate implementation;
- `examples/Printer/PrintParse.c` is an example instantiation that simply prints out the IR;
- tblgen/
  Helpers to bootstrap dialect parser specification.

### TODO

- [x] Bytecode structure callbacks for parsing all sections
- [ ] Complete Builtin dialect parsing methods
- [ ] Unit tests
- [ ] C++ example
- [ ] Switch to passing struct of callbacks for parsing in similar parsing methods
- [ ] Fuzzer
- [ ] Ensure unhandled is properly handled
- [ ] Reduce duplication with C++ side


