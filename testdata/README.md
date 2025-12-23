# mlirbcc Test Data

This directory contains MLIR bytecode files for testing the mlirbcc parser.

## Test Files

### Version 0 (Base Format)
These files were generated with earlier MLIR versions:

| File | Description |
|------|-------------|
| `general.mlirbc` | General MLIR operations and attributes |
| `blockargs.mlirbc` | Block arguments with locations |
| `resources.mlirbc` | Resource blob handling |
| `isolated_regions.mlirbc` | Isolated regions |
| `tosa_ops.mlirbc` | TOSA dialect operations |

### Version 6 (Current Format)
Generated with LLVM/MLIR from `llvm-project/build/bin/mlir-opt`:

| File | Source | Description |
|------|--------|-------------|
| `general_v6.mlirbc` | `general.mlir` | Same content, v6 format with properties |
| `blockargs_v6.mlirbc` | `blockargs.mlir` | Block args with v4+ location elision |

### Multi-Version Test (versions 0-6)
Files generated at each bytecode version for compatibility testing:

| File | Version | Key Features Tested |
|------|---------|---------------------|
| `version_test_v0.mlirbc` | 0 | Base format, attributes in dict |
| `version_test_v1.mlirbc` | 1 | Dialect versioning support |
| `version_test_v2.mlirbc` | 2 | Lazy loading (inline IR sections) |
| `version_test_v3.mlirbc` | 3 | Use-list ordering |
| `version_test_v4.mlirbc` | 4 | Block arg location elision |
| `version_test_v5.mlirbc` | 5 | Native properties encoding |
| `version_test_v6.mlirbc` | 6 | Full ODS properties support |

**Expected Differences:**
- v0-v4: Operations show attributes as `{ "name" = value; }`
- v5-v6: Operations show properties as `<props:N>`, attributes moved to properties section

## Generating New Test Files

To generate bytecode at the current version:
```sh
mlir-opt --emit-bytecode input.mlir -o output.mlirbc
```

To generate at a specific version:
```sh
mlir-opt --emit-bytecode --emit-bytecode-version=N input.mlir -o output.mlirbc
```

## Source Files

| File | Purpose |
|------|---------|
| `general.mlir` | General operations, regions, blocks |
| `blockargs.mlir` | Block arguments with typed locations |
| `version_test.mlir` | Simple func for multi-version testing |

