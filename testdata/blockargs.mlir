module {
  module {
    "bytecode.test1"() ({
        "bytecode.empty"() : () -> ()
        test.graph_region {
          %3:3 = "bytecode.results"() : () -> (i32, i64, i32)
          "bytecode.operands"(%3#0, %3#1, %3#2) : (i32, i64, i32) -> ()
          %4:2 = "bytecode.results"() : () -> (i32, i64)
          %5:2 = "bytecode.results"() : () -> (i32, i64)
        }
        "bytecode.branch"()[^bb1] : () -> ()
      ^bb1(%0: i32, %1: !bytecode.int):  // pred: ^bb0
        "bytecode.regions2"() ({
          ^bb1:
          %3:3 = "bytecode.results2"() : () -> (i32, i64, i32)
          "bytecode.operands4"(%0, %1) : (i32, !bytecode.int) -> ()
          ^bb2:
          %4:2 = "bytecode.results3"() : () -> (i32, i64)
          "bytecode.return"() : () -> ()
        }) : () -> ()
        "bytecode.return"() : () -> ()
      ^bb2(%2: i32, %3: i32, %4: i32): 
        "bytecode.return"() : () -> ()
    }) : () -> ()
  }
}

