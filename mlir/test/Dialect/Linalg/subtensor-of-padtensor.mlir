// RUN: mlir-opt %s -test-linalg-transform-patterns=test-swap-subtensor-padtensor -canonicalize  -split-input-file | FileCheck %s

// CHECK-LABEL: @static_data_only(
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<4x5xf32>
//       CHECK:   %[[RESULT:.*]] = tensor.extract_slice %[[ARG0]][1, 2] [2, 1] [1, 1] : tensor<4x5xf32> to tensor<2x1xf32>
//       CHECK:   return %[[RESULT]]
func @static_data_only(%arg0 : tensor<4x5xf32>, %pad : f32)
    -> tensor<2x1xf32> {
  %0 = linalg.pad_tensor %arg0 low[0, 0] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      linalg.yield %pad : f32
    } : tensor<4x5xf32> to tensor<11x13xf32>
  %1 = tensor.extract_slice %0[1, 2] [2, 1] [1, 1] : tensor<11x13xf32> to tensor<2x1xf32>
  return %1 : tensor<2x1xf32>
}

// -----

// CHECK-LABEL: @static_high_pad_only
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<4x5xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   linalg.pad_tensor
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   %[[RESULT:.*]] = tensor.generate
//       CHECK:     tensor.yield %[[PAD]]
//       CHECK:   return %[[RESULT]] : tensor<2x4xf32>
func @static_high_pad_only(%arg0 : tensor<4x5xf32>, %pad : f32)
    -> tensor<2x4xf32> {
  %0 = linalg.pad_tensor %arg0 low[0, 0] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      linalg.yield %pad : f32
    } : tensor<4x5xf32> to tensor<11x13xf32>
  %1 = tensor.extract_slice %0[4, 5] [2, 4] [1, 1] : tensor<11x13xf32> to tensor<2x4xf32>
  return %1 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: @static_low_pad_only
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<4x5xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   linalg.pad_tensor
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   %[[RESULT:.*]] = tensor.generate
//       CHECK:     tensor.yield %[[PAD]]
//       CHECK:   return %[[RESULT]] : tensor<2x3xf32>
func @static_low_pad_only(%arg0 : tensor<4x5xf32>, %pad : f32)
    -> tensor<2x3xf32> {
  %0 = linalg.pad_tensor %arg0 low[3, 7] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      linalg.yield %pad : f32
    } : tensor<4x5xf32> to tensor<14x20xf32>
  %1 = tensor.extract_slice %0[1, 3] [2, 3] [1, 1] : tensor<14x20xf32> to tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @static_low_pad_only_2
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<4x5xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   linalg.pad_tensor
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   %[[RESULT:.*]] = tensor.generate
//       CHECK:     tensor.yield %[[PAD]]
//       CHECK:   return %[[RESULT]] : tensor<1x3xf32>
func @static_low_pad_only_2(%arg0 : tensor<4x5xf32>, %pad : f32)
    -> tensor<1x3xf32> {
  %0 = linalg.pad_tensor %arg0 low[3, 7] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      linalg.yield %pad : f32
    } : tensor<4x5xf32> to tensor<14x20xf32>
  %1 = tensor.extract_slice %0[1, 3] [1, 3] [1, 1] : tensor<14x20xf32> to tensor<1x3xf32>
  return %1 : tensor<1x3xf32>
}

// -----

// CHECK-LABEL: @static_mixed_data_high_pad
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<4x5xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   linalg.pad_tensor
//       CHECK:   %[[SUBTENSOR:.*]] = tensor.extract_slice %[[ARG0]][2, 4] [2, 1] [1, 1] : tensor<4x5xf32> to tensor<2x1xf32>
//       CHECK:   %[[RESULT:.*]] = linalg.pad_tensor %[[SUBTENSOR]] low[0, 0] high[1, 3]
//       CHECK:     linalg.yield %[[PAD]]
//       CHECK:   return %[[RESULT]] : tensor<3x4xf32>
func @static_mixed_data_high_pad(%arg0 : tensor<4x5xf32>, %pad : f32)
    -> tensor<3x4xf32> {
  %0 = linalg.pad_tensor %arg0 low[0, 0] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      linalg.yield %pad : f32
    } : tensor<4x5xf32> to tensor<11x13xf32>
  %1 = tensor.extract_slice %0[2, 4] [3, 4] [1, 1] : tensor<11x13xf32> to tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: @static_mixed_data_low_pad
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<4x5xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   linalg.pad_tensor
//       CHECK:   %[[SUBTENSOR:.*]] = tensor.extract_slice %[[ARG0]][0, 0] [2, 1] [1, 1] : tensor<4x5xf32> to tensor<2x1xf32>
//       CHECK:   %[[RESULT:.*]] = linalg.pad_tensor %[[SUBTENSOR]] low[1, 3] high[0, 0]
//       CHECK:     linalg.yield %[[PAD]]
//       CHECK:   return %[[RESULT]] : tensor<3x4xf32>
func @static_mixed_data_low_pad(%arg0 : tensor<4x5xf32>, %pad : f32)
    -> tensor<3x4xf32> {
  %0 = linalg.pad_tensor %arg0 low[3, 7] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      linalg.yield %pad : f32
    } : tensor<4x5xf32> to tensor<14x20xf32>
  %1 = tensor.extract_slice %0[2, 4] [3, 4] [1, 1] : tensor<14x20xf32> to tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: @static_mixed_data_low_high_pad
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<4x5xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   linalg.pad_tensor
//       CHECK:   %[[RESULT:.*]] = linalg.pad_tensor %[[ARG0]] low[1, 1] high[2, 3]
//       CHECK:     linalg.yield %[[PAD]]
//       CHECK:   return %[[RESULT]] : tensor<7x9xf32>
func @static_mixed_data_low_high_pad(%arg0 : tensor<4x5xf32>, %pad : f32)
    -> tensor<7x9xf32> {
  %0 = linalg.pad_tensor %arg0 low[2, 3] high[7, 8] {
    ^bb0(%arg1: index, %arg2: index):
      linalg.yield %pad : f32
    } : tensor<4x5xf32> to tensor<13x16xf32>
  %1 = tensor.extract_slice %0[1, 2] [7, 9] [1, 1] : tensor<13x16xf32> to tensor<7x9xf32>
  return %1 : tensor<7x9xf32>
}

// -----

// CHECK-LABEL: @dynamic_high_pad
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<?x5xf32>
//   CHECK-NOT:   linalg.pad_tensor
//       CHECK:   %[[C0:.*]] = constant 0 : index
//       CHECK:   memref.dim %[[ARG0]], %[[C0]]
//       CHECK:   %[[RESULT:.*]] = scf.if %{{.*}} -> (tensor<3x4xf32>) {
//       CHECK:     %[[GEN:.*]] = tensor.generate
//       CHECK:     scf.yield %[[GEN]]
//       CHECK:   } else {
//       CHECK:     %[[SUBTENSOR:.*]] = tensor.extract_slice %[[ARG0]][%{{.*}}, 4] [%{{.*}}, 1] [1, 1] : tensor<?x5xf32> to tensor<?x1xf32>
//       CHECK:     %[[PADTENSOR:.*]] = linalg.pad_tensor %[[SUBTENSOR]] low[0, 0] high[%{{.*}}, 3]
//       CHECK:     %[[CAST:.*]] = tensor.cast %[[PADTENSOR]] : tensor<?x4xf32> to tensor<3x4xf32>
//       CHECK:     scf.yield %[[CAST]]
//       CHECK:   }
//       CHECK:   return %[[RESULT]]
func @dynamic_high_pad(%arg0 : tensor<?x5xf32>, %h1: index, %pad : f32) -> tensor<3x4xf32> {
  %0 = linalg.pad_tensor %arg0 low[0, 0] high[%h1, 8] {
    ^bb0(%arg1: index, %arg2: index):
      linalg.yield %pad : f32
    } : tensor<?x5xf32> to tensor<?x13xf32>
  %1 = tensor.extract_slice %0[2, 4] [3, 4] [1, 1] : tensor<?x13xf32> to tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

