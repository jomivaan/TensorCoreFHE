/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**
This example shows how to run matrix multiplication kernels using functions and data structures
provided by CUTLASS using tensor cores; which we run on a NVIDIA Volta GPU.

Writing a single high performance matrix multiplication kernel is hard but do-able. Whereas writing
high performance kernels at scale which works for multiple problem sizes with good abstractions is
really hard. CUTLASS solves this problem by providing simplified abstractions to compose
multiple sections of gemm kernel. When used properly, the kernels can hit peak performance of GPU
easily.

CUTLASS divides a kernel into hierarchical composable sections. Which means, at each thread, warp
and thread-block level, they compute on their own tile-size with higher level of tile sizes being
composed from lower level ones. Multiple thread-tiles (tile size each thread computes) can be used
to form warp-tiles (tile size each warp computes) and multiple warp tiles can be used to compute
threadblock-tile (tile size computed by a threadblock).

In thie example, we split variable initialization into
1. Setting up data properties : describes how matrices are laid out in the memory and how the kernel
can view them (logical to physical mapping)
2. Setting up computation properties : describes how the above set matrices will be used to compute
output of matrix multiplication.

First, we setup the data types of matrices A, B, C and D along with alpha, beta as the equation for
GEMM is D = alpha * A * B + beta * C. In CUTLASS, the kernels first compute A * B and leaves the
rest of the computation to end of the kernel as alpha * X + beta * C is a simple element-wise
operation on X (A * B) and C. We call this as epilogue of kernel. Hence, we setup data types for
alpha and beta to be equal to ElementComputeEpilogue = float. As we want to MMA instructions on
Volta and they support only half-precision floating point (fp16 or half), we use data type for
elements in input matrix A and B as cutlass::half_t. Volta also supports accumulation of partial dot
product to fp32, which can store wider range of numbers, we use it as data type of output matrix
elements and accumulation. We convey this to CUTLASS kernel by initializing template variables
ElementAccumulator (float), ElementComputeEpilogue (float), ElementInputA (cutlass::half_t),
ElementInputB (cutlass::half_t), ElementOutput (float). Communicating just the data type is not
enough. As the data is laid out linearly in memory, we have to convey the layout of matrices. We do
that by initializing template variable LayoutInputA to column major cutlass variable, LayoutInputB
to row major and LayoutOutput to row major. Next, we setup rules to comptue alpha * X + beta * C
which is called epilogue of the kernel. We initialize template variable EpilogueOp, which takes the
data type of output ElementOutput (int32_t), the number of elements per vector memory access (16),
data type of accumulator (int32_t) and data type of computation of linear combination (alpha * X +
beta * C).

Now that we setup the properties of data, we have to setup properties of computation.

Second, we create template variables of tile sizes for thread-block, warp and mma-op to 128x128x32,
64x64x32, 8x8x4 (MxNxK) respectively. When passed to instantiate CUTLASS GEMM kernel, it internally
deduce the amount of threads needed per thread-block, amount of shared memory, storing data in
bank-conflict free manner, and ton of other variables required to compose, initialize and launch a
high performance GEMM kernel. This is the beauty of CUTLASS, it relieves developer from
understanding and coding complicated hardware optimizations which can easily go wrong.

CUTLASS also supports multiple MMA pipelines in a CTA. What are MMA pipelines? MMA pipelines
constitute the whole process of loading input data from global memory to shared memory, loading data
from shared memory to registers, doing matrix multiplication, store to global memory. The below flow
sequence shows a typical mma pipeline.

matrix in global memory -> registers -> tile in shared memory -> registers -> mma -> registers ->
output to global memory

The problem with single pipeline is, each stage is synchronous which means, each stage has to wait
until the previous finished executing. There are stages in the pipeline which do not have fixed
latency, for example, the loads from global memory and shared memory. Therefore, we can add one more
pipeline with a phase shift in mma kernel to hide latency from global and shared memory loads.
Finally, the pipeline in a kernel looks like

(1) matrix in global memory -> (2) registers -> (3) tile in shared memory -> (4) registers -> (5)
mma -> (6) registers -> (7) output to global memory (1) <null> -> (2) <null> -> (3) matrix in global
memory -> (4) registers -> (5) tile in shared memory -> (6) registers -> (7) mma -> (8) registers ->
(9) output to global memory

This way, you can hide the second global memoroy load latency by doing computation on already loaded
input data.

There are few more template variables initialized such as, which threadblock tile of output matrix
is done which threadblock launched on an SM, CUDA SM architecture of GPU you want to run on.

These are all put together to create a template variable which describes CUTLASS GEMM kernel using
cutlass::gemm::device::Gemm template.

The next step is to initialize physical data, instantiate and initialize CUTLASS kernel and run it.
We use CUTLASS utilities to initialize, fill, compare matrices as they are simple and doesn't come
in the way of learning CUTLASS.

Once all the matrices are initialized and filled with data, create arguments tuple to launch CUTLASS
kernel which takes problem size (M = 5120, N = 4096 and K = 4096), matrices, alpha, beta and the
important one, split k-dimension factor. Along with that, we query CUTLASS if any scratch-space
memory required by the kernel we instantiated. If yes, we create it and pass it along with other
arguments created to initialize CUTLASS kernel then, the kernel is launched.

In this example, we later on launch a reference gemm kernel (from CUTLASS utilities) to compare if
the output from CUTLASS kernel is same as reference GEMM kernel.
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

using ElementAccumulator = int32_t;                  
using ElementComputeEpilogue = ElementAccumulator;  
using ElementInputA = uint8_t;             
using ElementInputB = uint8_t;             
using ElementOutput = int32_t;                        


using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;


using MMAOp = cutlass::arch::OpClassTensorOp;

using SmArch = cutlass::arch::Sm80;

using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 128>;  

using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 128>; 

using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 32>; 

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- this is the number of elements per
                                                       // vectorized memory access. For half
                                                       // precision, it's 8 elements. This becomes
                                                       // the vector width of math instructions in
                                                       // epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

constexpr int NumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;







__global__ void split_32_8_device(uint32_t *all_32b, 
                            uint8_t *first_8,
                            uint8_t *second_8,
                            uint8_t *third_8,
                            uint8_t *fourth_8){

  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  
  first_8[idx] = (uint8_t)(all_32b[idx] & 0xFF );
  second_8[idx] = (uint8_t)((all_32b[idx] >> 8) & 0xFF );
  third_8[idx] = (uint8_t)((all_32b[idx] >> 16) & 0xFF );
  fourth_8[idx] = (uint8_t)((all_32b[idx] >> 24) & 0xFF );

}


__global__ void merge_8_32_device( uint16_t *tensor_16b_1, 
                              uint16_t *tensor_16b_2,
                              uint16_t *tensor_16b_3,
                              uint16_t *tensor_16b_4,
                              uint32_t *result_32b){
  
  int idx = blockDim.x*blockIdx.x + threadIdx.x;

  result_32b[idx] = 0;
  result_32b[idx] = (uint32_t)tensor_16b_1[idx];
  result_32b[idx] += (((uint32_t)tensor_16b_2[idx]) << 8);
  result_32b[idx] += ((uint32_t)tensor_16b_3[idx] << 16);
  result_32b[idx] += ((uint32_t)tensor_16b_4[idx] << 24);


}


int main() {
  int N = 32768;
  int N1 = 256;
  int N2 = 128;

  /*INPUT*/
  using ElementInput = uint32_t; 
  using LayoutInput = cutlass::layout::RowMajor;

  cutlass::HostTensor<ElementInput, LayoutInput> tensor_input({N1,N2});
  for (int i = 0; i < N1; ++i) {
      for (int j = 0; j < N2; ++j) {

        
        tensor_input.host_ref().at({i, j}) = 255;
      } 
    }
  tensor_input.sync_device();

  cutlass::HostTensor<uint8_t, cutlass::layout::RowMajor> tensor_input_8b_1({N1,N2});
  cutlass::HostTensor<uint8_t, cutlass::layout::RowMajor> tensor_input_8b_2({N1,N2});
  cutlass::HostTensor<uint8_t, cutlass::layout::RowMajor> tensor_input_8b_3({N1,N2});
  cutlass::HostTensor<uint8_t, cutlass::layout::RowMajor> tensor_input_8b_4({N1,N2});

  dim3 grid_input(N1, 1, 1);
  dim3 block_input(N2, 1, 1);

  split_32_8_device<<< grid_input, block_input >>> (tensor_input.device_data(), 
                              tensor_input_8b_1.device_data(), 
                              tensor_input_8b_2.device_data(), 
                              tensor_input_8b_3.device_data(), 
                              tensor_input_8b_4.device_data());
  
  tensor_input_8b_1.sync_host();
  tensor_input_8b_2.sync_host();
  tensor_input_8b_3.sync_host();
  tensor_input_8b_4.sync_host();

  /*W1*/
  cutlass::HostTensor<ElementInput, LayoutInput> tensor_W1({N1,N1});
  for (int i = 0; i < N1; ++i) {
      for (int j = 0; j < N1; ++j) {
        tensor_W1.host_ref().at({i, j}) = 255;
      } 
    }
  tensor_W1.sync_device();

  cutlass::HostTensor<uint8_t, cutlass::layout::ColumnMajor> tensor_W1_8b_1({N1,N2});
  cutlass::HostTensor<uint8_t, cutlass::layout::ColumnMajor> tensor_W1_8b_2({N1,N2});
  cutlass::HostTensor<uint8_t, cutlass::layout::ColumnMajor> tensor_W1_8b_3({N1,N2});
  cutlass::HostTensor<uint8_t, cutlass::layout::ColumnMajor> tensor_W1_8b_4({N1,N2});

  dim3 grid_W1(N1, 1, 1);
  dim3 block_W1(N1, 1, 1);

  split_32_8_device<<< grid_W1, block_W1 >>> (tensor_W1.device_data(), 
                              tensor_W1_8b_1.device_data(), 
                              tensor_W1_8b_2.device_data(), 
                              tensor_W1_8b_3.device_data(), 
                              tensor_W1_8b_4.device_data());
  
  tensor_W1_8b_1.sync_host();
  tensor_W1_8b_2.sync_host();
  tensor_W1_8b_3.sync_host();
  tensor_W1_8b_4.sync_host();

  cutlass::HostTensor<int32_t, cutlass::layout::RowMajor> tensor_output_32b_1({N1,N2});

   
  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1); //leave as 1
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;
  cutlass::gemm::GemmCoord problem_size(N1, N2, N1);

  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                    tensor_input_8b_1.device_ref(),  // <- reference to matrix A on device
                                    tensor_W1_8b_1.device_ref(),  // <- reference to matrix B on device
                                    tensor_output_32b_1.device_ref(),  // <- reference to matrix C on device
                                    tensor_output_32b_1.device_ref(),  // <- reference to matrix D on device
                                    {alpha, beta},          // <- tuple of alpha and beta
                                    split_k_slices};        // <- k-dimension split factor

  // // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // // Check the problem size is supported or not 
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // // Launch initialized CUTLASS kernel
  status = gemm_op();
  CUTLASS_CHECK(status);

  tensor_output_32b_1.sync_host();
  std::cout << tensor_output_32b_1.host_view().at({0,0}) << std::endl;
  

}

// split_32_8(tensor_input.device_data(), N1,N2);


//   cutlass::HostTensor<uint16_t, LayoutInput> tensor_16b_1({N1,N2});
//   cutlass::HostTensor<uint16_t, LayoutInput> tensor_16b_2({N1,N2});
//   cutlass::HostTensor<uint16_t, LayoutInput> tensor_16b_3({N1,N2});
//   cutlass::HostTensor<uint16_t, LayoutInput> tensor_16b_4({N1,N2});
//   cutlass::reference::host::TensorFill(tensor_16b_1.host_view()); 
//   cutlass::reference::host::TensorFill(tensor_16b_2.host_view()); 
//   cutlass::reference::host::TensorFill(tensor_16b_3.host_view()); 
//   cutlass::reference::host::TensorFill(tensor_16b_4.host_view()); 
//   tensor_16b_1.host_ref().at({0, 0}) = 1;
//   tensor_16b_2.host_ref().at({0, 0}) = 1;
//   tensor_16b_3.host_ref().at({0, 0}) = 1;
//   tensor_16b_4.host_ref().at({0, 0}) = 1;

//   tensor_16b_1.sync_device();
//   tensor_16b_2.sync_device();
//   tensor_16b_3.sync_device();
//   tensor_16b_4.sync_device();
//   merge_8_32(tensor_16b_1.device_data(), tensor_16b_2.device_data(), tensor_16b_3.device_data(), tensor_16b_4.device_data(), N1, N2);