// MLIR for version testing - exercises features from each bytecode version
module {
  // v4: Block arg location elision - some args have unknown locations
  func.func @test_block_arg_locations(%arg0: i32 loc(unknown), %arg1: i64 loc("named")) -> i32 {
    %c1 = arith.constant 1 : i32
    %0 = arith.addi %arg0, %c1 : i32
    cf.br ^bb1(%0 : i32)
  ^bb1(%blockarg: i32 loc(unknown)):  // Block arg with unknown location
    return %blockarg : i32
  }
  
  // v2: Isolated region for lazy loading (using func as it's isolated from above)
  func.func @nested_func() {
    %x = arith.constant 42 : i32
    return
  }
}
