module {
  module {
    "bytecode.test1"() ({
      "bytecode.empty"() : () -> ()
      "bytecode.attributes"() {attra = 10 : i64, attrb = #bytecode.attr} : () -> ()
      test.graph_region {
        "bytecode.operands"(%3#0, %3#1, %3#2) : (i32, i64, i32) -> ()
        %3:3 = "bytecode.results"() : () -> (i32, i64, i32)
      }
      "bytecode.branch"()[^bb1] : () -> ()
    ^bb1(%0: i32, %1: !bytecode.int, %2: !pdl.operation):  // pred: ^bb0
      "bytecode.regions"() ({
        "bytecode.operands"(%0, %1, %2) : (i32, !bytecode.int, !pdl.operation) -> ()
        "bytecode.return"() : () -> ()
      }, {
        "bytecode.operands2"(%0, %1, %2) : (i32, !bytecode.int, !pdl.operation) -> ()
        "bytecode.operands3"(%0, %1, %2) : (i32, !bytecode.int, !pdl.operation) -> ()
        "bytecode.return"() : () -> ()
      }) : () -> ()
      "bytecode.regions2"() ({
        "bytecode.operands4"(%0, %1) : (i32, !bytecode.int) -> ()
        "bytecode.return"() : () -> ()
      }) : () -> ()
      "bytecode.return"() : () -> ()
    }) : () -> ()
  }
  module {
    test.graph_region {
      "bytecode.operands"(%0#0, %0#1, %0#2) : (i32, i64, i32) -> ()
      %0:3 = "bytecode.results"() : () -> (i32, i64, i32)
    }
  }
}

{-#
  external_resources: {
    external: {
      blobE: "0x08000000010000000000000002000000000000000300000000000000",
      boolE: true,
      stringE: "string"
    }
  }
#-}

