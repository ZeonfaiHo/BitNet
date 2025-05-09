#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <mma.h>
#include <iostream>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1
#else
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 0
#endif

#define uchar unsigned char

template <typename T1, typename T2>
__device__ void decode_i1s_to_i8s_l16(T1 *_i1s, T2 *_i8s, const int N = 16)
{
  int *i8s = reinterpret_cast<int *>(_i8s);
  int16_t i1s_i16 = *reinterpret_cast<int16_t *>(_i1s);
  // permutate: {e0,e4,e8,e12,e2,e6,e10,e14,e1,e5,e9,e13,e3,e7,e11,e15}
  // into: {e0,e4,e8,e12,x,x,x,x,e1,e5,e9,x,x,x,x,e13,e2,e6,e10,e14,e1,e5,e9,e13,e3,e7,e11,e15,x,x,x,x}
  int i1s = (i1s_i16 & 0x0f0f);
  i1s |= ((i1s_i16 & 0xf0f0) << 12); 
  // i1s        {0..,e15,e14,e13,e12,e11,e10,e9,e8,e7,e6,e5,e4,e3,e2,e1,e0}
  // interleave {0..,e15,e13,e11,e9,e7,e5,e3,e1,e14,e12,e10,e8,e6,e4,e2,e0}
  // First, we extract the i1s and construct an intermediate fp16 number.
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa; // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x01010101;      // 0x1 -> 0b01 select 0,1
  static constexpr uint I8s_MAGIC_NUM = 0x00000000;

  for (int i = 0; i < N / 4; i++)
  {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(i8s[i])
                 : "r"(i1s >> i), "n"(BOTTOM_MASK), "n"(I8s_MAGIC_NUM), "n"(immLut));
  }
}


template <typename T1, typename T2>
__device__ void decode_i2s_to_i8s(T1 *_i2s, T2 *_i8s, const int N = 16)
{
  // convert 8 int2b_t to 8 int8b_t -> 2 int32
  uint *i8s = reinterpret_cast<uint *>(_i8s);

  // i2s = {e7,e6,e5,e4,e3,e2,e1,e0}
  // also require interleave {e7,e3,e6,e2,e5,e1,e4,e0}
  uint const i2s = *_i2s;

  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;     // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x03030303;          // 0xf -> 0b11 select 0,3
  static constexpr uint I4s_TO_I8s_MAGIC_NUM = 0x00000000; // 1024

#pragma unroll
  for (int i = 0; i < (N / 2); i++)
  {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(i8s[i])
                 : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
    i8s[i] = __vsubss4(i8s[i], 0x02020202);
  }
}


template <typename T1, typename T2>
__device__ void decode_i4s_to_i8s(T1 *_i4s, T2 *_i8s, const int N = 8)
{
  uint *i8s = reinterpret_cast<uint *>(_i8s);
  uint i4s = *_i4s;
  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;     // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x0f0f0f0f;          // 0xf -> 0b1111 select 0,4
  static constexpr uint I4s_TO_I8s_MAGIC_NUM = 0x00000000; // 1024

#pragma unroll
  for (int i = 0; i < (N / 4); i++)
  {
    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(i8s[i])
                 : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
  }
}

template <typename T1, typename T2>
__device__ void decode_i2s_to_i8s_adsbrain(T1 *_i2s, T2 *_i8s, const int N = 16)
{
  // convert 8 int2b_t to 8 int8b_t -> 2 int32
  uint *i8s = reinterpret_cast<uint *>(_i8s);

  // i2s = {e7,e6,e5,e4,e3,e2,e1,e0}
  // also require interleave {e7,e3,e6,e2,e5,e1,e4,e0}
  uint const i2s = *_i2s;

#pragma unroll
  for (int i = 0; i < (N / 2); i++)
  {
    i8s[i] = __byte_perm(0x0100ff00, 0x0100ff00, (i2s >> ((i % 2) * 2 + (i / 2) * 16))); 
  }
}

// #include "cutlass/cutlass.h"
// #include "cutlass/gemm/warp/mma_tensor_op.h"
// #include "cutlass/gemm/warp/mma_tensor_op_sm70.h"

// namespace cutlass {
// namespace gemm {
// namespace warp {

// template<class MmaWarp, int KSize>
// class MMAWarpWrapper {
// public:
//   typename MmaWarp::FragmentA frag_A[2];
//   typename MmaWarp::FragmentB frag_B[2];
//   typename MmaWarp::FragmentC accum;
//   MmaWarp mma_op;
//   typename MmaWarp::IteratorA iter_A;
//   typename MmaWarp::IteratorB iter_B;
//   const int warp_idx_m_, warp_idx_n_, lane_id_;

//   using TensorRefA = typename MmaWarp::IteratorA::TensorRef;
//   using TensorRefB = typename MmaWarp::IteratorB::TensorRef;
//   static_assert(KSize % MmaWarp::Shape::kK == 0);
//   static int constexpr kKgroups = KSize / MmaWarp::Shape::kK;

//   CUTLASS_DEVICE
//   MMAWarpWrapper(int warp_idx_m, int warp_idx_n, int lane_id)
//   : warp_idx_m_(warp_idx_m), warp_idx_n_(warp_idx_n), lane_id_(lane_id), iter_A({nullptr, 0}, 0), iter_B({nullptr, 0}, 0) {
//     accum.clear();
//   }

//   CUTLASS_DEVICE
//   void prologue(const TensorRefA &ref_A, const TensorRefB &ref_B) {
//     iter_A = typename MmaWarp::IteratorA(ref_A, lane_id_);
//     iter_B = typename MmaWarp::IteratorB(ref_B, lane_id_);
//     iter_A.add_tile_offset({warp_idx_m_, 0});
//     iter_B.add_tile_offset({0, warp_idx_n_});
//     iter_A.load(frag_A[0]);
//     iter_B.load(frag_B[0]);
//     ++iter_A;
//     ++iter_B;
//   }
//   CUTLASS_DEVICE
//   void body() {
//     CUTLASS_PRAGMA_UNROLL
//     for (int k = 0; k < kKgroups - 1; ++k) {
//       iter_A.load(frag_A[(k + 1) % 2]);
//       iter_B.load(frag_B[(k + 1) % 2]);
//       ++iter_A;
//       ++iter_B;
//       mma_op(accum, frag_A[k % 2], frag_B[k % 2], accum);
//     }
//     __syncthreads();
//   }
//   CUTLASS_DEVICE
//   void epilogue() {
//     mma_op(accum, frag_A[(kKgroups - 1) % 2], frag_B[(kKgroups - 1) % 2], accum);
//   }
// };

// template <
//   typename Shape,
//   typename SMemLayoutA,
//   typename SMemLayoutB
// >
// class GemmTensorOp {
// public:
//   using InstructionShape = GemmShape<16, 8, 16>;
//   using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
//     cutlass::arch::Mma<
//       InstructionShape,
//       32,
//       cutlass::half_t,
//       cutlass::layout::RowMajor,
//       cutlass::half_t,
//       cutlass::layout::ColumnMajor,
//       cutlass::half_t,
//       cutlass::layout::RowMajor,
//       cutlass::arch::OpMultiplyAdd
//     >,
//     cutlass::MatrixShape<1, 1>
//   >;

//   using MmaWarp = typename cutlass::gemm::warp::MmaTensorOp<
//     GemmShape<Shape::kM, Shape::kN, InstructionShape::kK>,
//     cutlass::half_t,
//     SMemLayoutA,
//     cutlass::half_t,
//     SMemLayoutB,
//     cutlass::half_t,
//     cutlass::layout::RowMajor,
//     Policy
//   >;
//   using MMA = MMAWarpWrapper<MmaWarp, Shape::kK>;
//   MMA mma;

//   CUTLASS_DEVICE
//   GemmTensorOp(int warp_idx_m, int warp_idx_n, int lane_id)
//   : mma(warp_idx_m, warp_idx_n, lane_id) {}
//   CUTLASS_DEVICE
//   half& operator[](size_t i) const {
//     return ((half*)mma.accum.data())[i];
//   }
//   CUTLASS_DEVICE
//   half* operator+(size_t i) const {
//     return (half*)mma.accum.data() + i;
//   }
// };

// template <
//   typename Shape,
//   typename SMemLayoutA,
//   typename LayoutA,
//   typename SMemLayoutB,
//   typename LayoutB,
//   typename LayoutC
// >
// class VoltaGemmTensorOp {
// public:
//   using InstructionShape = GemmShape<16, 16, 4>;
//   using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
//     cutlass::arch::Mma<
//       InstructionShape,
//       32,
//       cutlass::half_t,
//       LayoutA,
//       cutlass::half_t,
//       LayoutB,
//       cutlass::half_t,
//       LayoutC,
//       cutlass::arch::OpMultiplyAdd
//     >,
//     cutlass::MatrixShape<1, 1>
//   >;

//   using MmaWarp = typename cutlass::gemm::warp::MmaVoltaTensorOp<
//     GemmShape<Shape::kM, Shape::kN, InstructionShape::kK>,
//     cutlass::half_t,
//     SMemLayoutA,
//     cutlass::half_t,
//     SMemLayoutB,
//     cutlass::half_t,
//     LayoutC,
//     Policy
//   >;
//   using MMA = MMAWarpWrapper<MmaWarp, Shape::kK>;
//   MMA mma;

//   CUTLASS_DEVICE
//   VoltaGemmTensorOp(int warp_idx_m, int warp_idx_n, int lane_id)
//   : mma(warp_idx_m, warp_idx_n, lane_id) {}
//   CUTLASS_DEVICE
//   half& operator[](size_t i) const {
//     return ((half*)mma.accum.data())[i];
//   }
//   CUTLASS_DEVICE
//   half* operator+(size_t i) const {
//     return (half*)mma.accum.data() + i;
//   }
// };

// template <
//   typename Shape,
//   typename SMemLayoutA,
//   typename SMemLayoutB
// >
// class GemmI8TensorOp {
// public:
//   using InstructionShape = GemmShape<16, 8, 32>;
//   using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
//     cutlass::arch::Mma<
//       InstructionShape,
//       32,
//       int8_t,
//       cutlass::layout::RowMajor,
//       int8_t,
//       cutlass::layout::ColumnMajor,
//       int,
//       cutlass::layout::RowMajor,
//       cutlass::arch::OpMultiplyAdd
//     >,
//     cutlass::MatrixShape<1, 1>
//   >;

//   using MmaWarp = typename cutlass::gemm::warp::MmaTensorOp<
//     GemmShape<Shape::kM, Shape::kN, InstructionShape::kK>,
//     int8_t,
//     SMemLayoutA,
//     int8_t,
//     SMemLayoutB,
//     int,
//     cutlass::layout::RowMajor,
//     Policy
//   >;
//   using MMA = MMAWarpWrapper<MmaWarp, Shape::kK>;
//   MMA mma;

//   CUTLASS_DEVICE
//   GemmI8TensorOp(int warp_idx_m, int warp_idx_n, int lane_id)
//   : mma(warp_idx_m, warp_idx_n, lane_id) {}
//   CUTLASS_DEVICE
//   int& operator[](size_t i) const {
//     return ((int*)mma.accum.data())[i];
//   }
//   CUTLASS_DEVICE
//   int* operator+(size_t i) const {
//     return (int*)mma.accum.data() + i;
//   }
// };

// }}}

// template<class TensorOp>
// CUTLASS_DEVICE void call_cutlass_mma_body(TensorOp& op) {
//   op.mma.body();
// }

// template<class TensorOp>
// CUTLASS_DEVICE void call_cutlass_mma_epilogue(TensorOp& op) {
//   op.mma.epilogue();
// }

// template<class TensorOp>
// CUTLASS_DEVICE void call_cutlass_mma_prologue(TensorOp& op, void* pA, void* pB, int sA, int sB) {
//   using TensorRefA = typename TensorOp::MMA::TensorRefA;
//   using TensorRefB = typename TensorOp::MMA::TensorRefB;
//   TensorRefA refA{(typename TensorRefA::Element*)pA, sA};
//   TensorRefB refB{(typename TensorRefB::Element*)pB, sB};
//   op.mma.prologue(refA, refB);
// }


template <int M>
__global__ void __launch_bounds__(125) ladder_int8xint2_1_4800_3200_kernel(int8_t* __restrict__ A, int8_t* __restrict__ B, __nv_bfloat16* __restrict__ D, __nv_bfloat16* __restrict__ s, __nv_bfloat16* __restrict__ ws) {

  const int IN_DIM = 3200;
  const int OUT_DIM = 4800;
  int in_thread_C_local[M];
  int B_local[1];
  signed char B_decode_local[16];
  signed char A_local[16 * M];
  __shared__ int red_buf0[125 * M];
  #pragma unroll
  for (int i = 0; i < M; i++) {
    in_thread_C_local[i] = 0;
  }
  #pragma unroll
  for (int ax1_0 = 0; ax1_0 < 8; ++ax1_0) {
    B_local[0] = *(int*)(B + ((((((int)blockIdx.x) * 4000) + (((int)threadIdx.y) * 800)) + (ax1_0 * 100)) + (((int)threadIdx.x) * 4)));
    decode_i2s_to_i8s_adsbrain(B_local, B_decode_local, 16);
    #pragma unroll
    for (int i = 0; i < M; i++) {
      *(int4*)(A_local + (i * 16)) = *(int4*)(A + ((ax1_0 * 400) + (((int)threadIdx.x) * 16)) + IN_DIM * i);
    }
    #pragma unroll
    for (int ax1_2_0 = 0; ax1_2_0 < 4; ++ax1_2_0) { 
      #pragma unroll
      for (int i = 0; i < M; i++) {
        in_thread_C_local[i] = __dp4a(*(int *)&A_local[((ax1_2_0 * 4) + i * 16)], *(int *)&B_decode_local[((ax1_2_0 * 4))], in_thread_C_local[i]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < M; i++) {
    ((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] = in_thread_C_local[i];
  }
  __syncthreads();
  if (((int)threadIdx.x) < 9) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + 16) + i * 125]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 8) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + 8) + i * 125]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 4) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + 4) + i * 125]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 2) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + 2) + i * 125]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 1) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + 1) + i * 125]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    int out_idx = ((((int)blockIdx.x) * 5) + ((int)threadIdx.y));
    int ws_idx = 2 - min((4799 - out_idx)/800, 2);
    D[out_idx] = (__nv_bfloat16)(((float)((volatile int*)red_buf0)[(((int)threadIdx.y) * 25)])/(float)s[0]*(float)ws[ws_idx]);
    #pragma unroll
    for (int i = 0; i < M; i++) {
      D[out_idx + OUT_DIM * i] = (__nv_bfloat16)(((float)((volatile int*)red_buf0)[(((int)threadIdx.y) * 25) + i * 125])/__bfloat162float(s[i])*__bfloat162float(ws[ws_idx]));
    }
  }
}


template <int M>
__global__ void __launch_bounds__(125) ladder_int8xint2_1_3200_3200_kernel(int8_t* __restrict__ A, int8_t* __restrict__ B, __nv_bfloat16* __restrict__ D, __nv_bfloat16* __restrict__ s, __nv_bfloat16* __restrict__ ws) {

  const int IN_DIM = 3200;
  const int OUT_DIM = 3200;
  int in_thread_C_local[M];
  int B_local[1];
  signed char B_decode_local[16];
  signed char A_local[16 * M];
  __shared__ int red_buf0[125 * M];
  #pragma unroll
  for (int i = 0; i < M; i++) {
    in_thread_C_local[i] = 0;
  }
  #pragma unroll
  for (int ax1_0 = 0; ax1_0 < 8; ++ax1_0) {
    B_local[0] = *(int*)(B + ((((((int)blockIdx.x) * 4000) + (((int)threadIdx.y) * 800)) + (ax1_0 * 100)) + (((int)threadIdx.x) * 4)));
    decode_i2s_to_i8s_adsbrain(B_local, B_decode_local, 16);
    #pragma unroll
    for (int i = 0; i < M; i++) {
      *(int4*)(A_local + (i * 16)) = *(int4*)(A + ((ax1_0 * 400) + (((int)threadIdx.x) * 16)) + IN_DIM * i);
    }
    #pragma unroll
    for (int ax1_2_0 = 0; ax1_2_0 < 4; ++ax1_2_0) {
      #pragma unroll
      for (int i = 0; i < M; i++) {
        in_thread_C_local[i] = __dp4a(*(int *)&A_local[((ax1_2_0 * 4) + i * 16)], *(int *)&B_decode_local[((ax1_2_0 * 4))], in_thread_C_local[i]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < M; i++) {
    ((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] = in_thread_C_local[i];
  }
  __syncthreads();
  if (((int)threadIdx.x) < 9) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + 16) + i * 125]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 8) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + 8) + i * 125]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 4) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + 4) + i * 125]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 2) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + 2) + i * 125]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 1) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + 1) + i * 125]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      D[((((int)blockIdx.x) * 5) + ((int)threadIdx.y)) + OUT_DIM * i] = (((float)((volatile int*)red_buf0)[(((int)threadIdx.y) * 25) + i * 125])/(float)s[i]*(float)ws[0]);
    }
    // D[((((int)blockIdx.x) * 2) + ((int)threadIdx.y))] = (half)((float)(red_result[((int)threadIdx.y)])/(float)s[0]*(float)ws[0]);
  }
}


// __global__ void __launch_bounds__(128) ladder_int8xint2_1_20480_3200_kernel(int8_t* __restrict__ A, int8_t* __restrict__ B, half* __restrict__ D, half* __restrict__ s, half* __restrict__ ws) {

//   int in_thread_C_local[1];
//   int B_local[1];
//   signed char B_decode_local[16];
//   signed char A_local[16];
//   int red_buf0[1];
//   in_thread_C_local[0] = 0;
//   for (int ax1_0 = 0; ax1_0 < 25; ++ax1_0) {
//     B_local[0] = *(int*)(B + ((((((int)blockIdx.x) * 12800) + (((int)threadIdx.y) * 800)) + (ax1_0 * 32)) + (((int)threadIdx.x) * 4)));
//     decode_i2s_to_i8s(B_local, B_decode_local, 16);
//     *(int4*)(A_local + 0) = *(int4*)(A + ((ax1_0 * 128) + (((int)threadIdx.x) * 16)));
//     for (int ax1_2_0 = 0; ax1_2_0 < 4; ++ax1_2_0) {
//       in_thread_C_local[0] = __dp4a(*(int *)&A_local[((ax1_2_0 * 4))],*(int *)&B_decode_local[((ax1_2_0 * 4))], in_thread_C_local[0]);
//     }
//   }
//   uint mask[1];
//   int t0[1];
//   red_buf0[0] = in_thread_C_local[0];
//   mask[0] = (__activemask() & ((uint)255 << ((uint)8 * ((uint)((int)threadIdx.y)))));
//   t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
//   red_buf0[0] = (red_buf0[0] + t0[0]);
//   t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
//   red_buf0[0] = (red_buf0[0] + t0[0]);
//   t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
//   red_buf0[0] = (red_buf0[0] + t0[0]);
//   red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 8), 32);
//   if (((int)threadIdx.x) == 0) {
//     int out_idx = ((((int)blockIdx.x) * 16) + ((int)threadIdx.y));
//     D[out_idx] = (half)(((float)red_buf0[0])/(float)s[0]*(float)ws[out_idx/10240]);
//     // D[((((int)blockIdx.x) * 16) + ((int)threadIdx.y))] = ((half)red_buf0[0]);
//   }
// }

template <int M>
__global__ void __launch_bounds__(125) ladder_int8xint2_1_20480_3200_kernel(int8_t* __restrict__ A, int8_t* __restrict__ B, __nv_bfloat16* __restrict__ D, __nv_bfloat16* __restrict__ s, __nv_bfloat16* __restrict__ ws) {

  const int IN_DIM = 3200;
  const int OUT_DIM = 20480;
  int in_thread_C_local[M];
  int B_local[1];
  signed char B_decode_local[16];
  signed char A_local[16 * M];
  __shared__ int red_buf0[125 * M];
  #pragma unroll
  for (int i = 0; i < M; i++) {
    in_thread_C_local[i] = 0;
  }
  #pragma unroll
  for (int ax1_0 = 0; ax1_0 < 8; ++ax1_0) {
    B_local[0] = *(int*)(B + ((((((int)blockIdx.x) * 4000) + (((int)threadIdx.y) * 800)) + (ax1_0 * 100)) + (((int)threadIdx.x) * 4)));
    decode_i2s_to_i8s_adsbrain(B_local, B_decode_local, 16);
    #pragma unroll
    for (int i = 0; i < M; i++) {
      *(int4*)(A_local + (i * 16)) = *(int4*)(A + ((ax1_0 * 400) + (((int)threadIdx.x) * 16) + IN_DIM * i));
    }
    #pragma unroll
    for (int ax1_2_0 = 0; ax1_2_0 < 4; ++ax1_2_0) {
      #pragma unroll
      for (int i = 0; i < M; i++) {
        in_thread_C_local[i] = __dp4a(*(int *)&A_local[((ax1_2_0 * 4) + i * 16)],  *(int *)&B_decode_local[((ax1_2_0 * 4))], in_thread_C_local[i]);
      }
    }
  }
  __syncthreads();
  // ((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x))] = in_thread_C_local[0];
  #pragma unroll
  for (int i = 0; i < M; i++) {
    ((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] = in_thread_C_local[i];
  }
  __syncthreads();
  if (((int)threadIdx.x) < 9) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + 16) + i * 125]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 8) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + 8) + i * 125]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 4) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + 4) + i * 125]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 2) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + 2) + i * 125]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 1) {    
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + i * 125] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * 25) + ((int)threadIdx.x)) + 1) + i * 125]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    // int out_idx = ((((int)blockIdx.x) * 5) + ((int)threadIdx.y));
    // D[out_idx] = (((float)((volatile int*)red_buf0)[(((int)threadIdx.y) * 25)])/(float)s[0]*(float)ws[out_idx/10240]);
    #pragma unroll
    for (int i = 0; i < M; i++) {
      D[((((int)blockIdx.x) * 5) + ((int)threadIdx.y)) + OUT_DIM * i] = (((float)((volatile int*)red_buf0)[(((int)threadIdx.y) * 25) + i * 125])/(float)s[i]*(float)ws[blockIdx.x/2048]);    
    }
    // D[((((int)blockIdx.x) * 5) + ((int)threadIdx.y))] = ((half)((volatile int*)red_buf0)[(((int)threadIdx.y) * 25)]);
  }
}


template <int M>
__global__ void __launch_bounds__(128) ladder_int8xint2_1_3200_10240_kernel(int8_t* __restrict__ A, int8_t* __restrict__ B, __nv_bfloat16* __restrict__ D, __nv_bfloat16* __restrict__ s, __nv_bfloat16* __restrict__ ws) {

  const int IN_DIM = 10240;
  const int OUT_DIM = 3200;
  int in_thread_C_local[M];
  int B_local[1];
  signed char B_decode_local[16];
  signed char A_local[16 * M];
  __shared__ int red_result[1 * M];
  #pragma unroll
  for (int i = 0; i < M; i++) {
    in_thread_C_local[i] = 0;
  }
  #pragma unroll
  for (int ax1_0 = 0; ax1_0 < 5; ++ax1_0) {
    B_local[0] = *(int*)(B + (((((int)blockIdx.x) * 2560) + (ax1_0 * 512)) + (((int)threadIdx.x) * 4)));
    decode_i2s_to_i8s_adsbrain(B_local, B_decode_local, 16);
    #pragma unroll
    for (int i = 0; i < M; i++) {
      *(int4*)(A_local + (i * 16)) = *(int4*)(A + ((ax1_0 * 2048) + (((int)threadIdx.x) * 16) + IN_DIM * i));
    }
    #pragma unroll
    for (int ax1_2_0 = 0; ax1_2_0 < 4; ++ax1_2_0) {
      #pragma unroll
      for (int i = 0; i < M; i++) {
        in_thread_C_local[i] = __dp4a(*(int *)&A_local[((ax1_2_0 * 4) + i * 16)], *(int *)&B_decode_local[((ax1_2_0 * 4))], in_thread_C_local[i]);
      }
    }
  }
  int red_buf0[M];
  uint mask[1];
  int t0[M];
  int red_buf0_1[M];
  uint mask_1[1];
  int t0_1[M];
  __shared__ int red_buf_staging[4 * M];
  #pragma unroll
  for (int i = 0; i < M; i++) {
    red_buf0_1[i] = in_thread_C_local[i];
  }
  mask_1[0] = __activemask();
  #pragma unroll
  for (int i = 0; i < M; i++) {
    t0_1[i] = __shfl_down_sync(mask_1[0], red_buf0_1[i], 16, 32);
    red_buf0_1[i] = (red_buf0_1[i] + t0_1[i]);
  }
  #pragma unroll
  for (int i = 0; i < M; i++) {
    t0_1[i] = __shfl_down_sync(mask_1[0], red_buf0_1[i], 8, 32);
    red_buf0_1[i] = (red_buf0_1[i] + t0_1[i]);
  }
  #pragma unroll
  for (int i = 0; i < M; i++) {
    t0_1[i] = __shfl_down_sync(mask_1[0], red_buf0_1[i], 4, 32);
    red_buf0_1[i] = (red_buf0_1[i] + t0_1[i]);
  }
  #pragma unroll
  for (int i = 0; i < M; i++) {
    t0_1[i] = __shfl_down_sync(mask_1[0], red_buf0_1[i], 2, 32);
    red_buf0_1[i] = (red_buf0_1[i] + t0_1[i]);
  }
  #pragma unroll
  for (int i = 0; i < M; i++) {
    t0_1[i] = __shfl_down_sync(mask_1[0], red_buf0_1[i], 1, 32);
    red_buf0_1[i] = (red_buf0_1[i] + t0_1[i]);
  }
  if ((((int)threadIdx.x) % 32) == 0) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      red_buf_staging[(((int)threadIdx.x) >> 5) + (i * 4)] = red_buf0_1[i];
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 4) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      red_buf0[i] = red_buf_staging[((int)threadIdx.x) + (i * 4)];
    }
  }
  mask[0] = (__activemask() & (uint)15);
  #pragma unroll
  for (int i = 0; i < M; i++) {
    t0[i] = __shfl_down_sync(mask[0], red_buf0[i], 2, 32);
    red_buf0[i] = (red_buf0[i] + t0[i]);
  }
  #pragma unroll
  for (int i = 0; i < M; i++) {
    t0[i] = __shfl_down_sync(mask[0], red_buf0[i], 1, 32);
    red_buf0[i] = (red_buf0[i] + t0[i]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    // D[((int)blockIdx.x)] = (half)(((float)red_buf0[0])/(float)s[0]*(float)ws[0]);
    #pragma unroll
    for (int i = 0; i < M; i++) {
      D[((int)blockIdx.x) + OUT_DIM * i] = (__nv_bfloat16)(((float)red_buf0[i])/(float)s[i]*(float)ws[0]);
    }
  }
}


