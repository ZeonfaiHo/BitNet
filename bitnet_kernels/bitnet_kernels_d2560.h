template <int M, int BlkDimX, int BlkDimY>
__global__ void __launch_bounds__(128) ladder_int8xint2_1_3840_2560_kernel(int8_t* __restrict__ A, int8_t* __restrict__ B, __nv_bfloat16* __restrict__ D, __nv_bfloat16* __restrict__ s, __nv_bfloat16* __restrict__ ws) {
  const int IN_DIM = 2560;
  const int OUT_DIM = 3840;
  const int LOOP_PER_THREAD = IN_DIM / BlkDimX / 16;
  const int BlkSize = BlkDimX * BlkDimY;
  int in_thread_C_local[M];
  int B_local[1];
  signed char B_decode_local[16];
  signed char A_local[16 * M];
  __shared__ int red_buf0[BlkSize * M];
  #pragma unroll
  for (int i = 0; i < M; i++) {
    in_thread_C_local[i] = 0;
  }
  #pragma unroll
  for (int ax1_0 = 0; ax1_0 < LOOP_PER_THREAD; ++ax1_0) {
    // B_local[0] = *(int*)(B + ((((((int)blockIdx.x) * 3200) + (((int)threadIdx.y) * 640)) + (ax1_0 * 128)) + (((int)threadIdx.x) * 4)));
    B_local[0] = *(int*)(B + ((((((int)blockIdx.x) * blockDim.y * IN_DIM / 4) + (((int)threadIdx.y) * IN_DIM / 4)) + (ax1_0 * IN_DIM / LOOP_PER_THREAD / 4)) + (((int)threadIdx.x) * 4)));
    decode_i2s_to_i8s_adsbrain(B_local, B_decode_local, 16);
    #pragma unroll
    for (int i = 0; i < M; i++) {
      // *(int4*)(A_local + (i * 16)) = *(int4*)(A + ((ax1_0 * 512) + (((int)threadIdx.x) * 16)) + IN_DIM * i);
      *(int4*)(A_local + (i * 16)) = *(int4*)(A + ((ax1_0 * IN_DIM / LOOP_PER_THREAD) + (((int)threadIdx.x) * 16)) + IN_DIM * i);
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
    ((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] = in_thread_C_local[i];
  }
  __syncthreads();
  if (((int)threadIdx.x) < BlkDimX - 16) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + 16) + i * BlkSize]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 8) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + 8) + i * BlkSize]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 4) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + 4) + i * BlkSize]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 2) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + 2) + i * BlkSize]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 1) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + 1) + i * BlkSize]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    int out_idx = ((((int)blockIdx.x) * blockDim.y) + ((int)threadIdx.y));
    int ws_idx = (out_idx < 2560 ? 0 : (out_idx < 3200 ? 1 : 2));
    #pragma unroll
    for (int i = 0; i < M; i++) {
      D[out_idx + OUT_DIM * i] = (((float)((volatile int*)red_buf0)[(((int)threadIdx.y) * blockDim.x) + i * BlkSize])/__bfloat162float(s[i])*__bfloat162float(ws[ws_idx]));
    }
  }
}

template <int M, int BlkDimX, int BlkDimY>
__global__ void __launch_bounds__(128) ladder_int8xint2_1_2560_2560_kernel(int8_t* __restrict__ A, int8_t* __restrict__ B, __nv_bfloat16* __restrict__ D, __nv_bfloat16* __restrict__ s, __nv_bfloat16* __restrict__ ws) {

  const int IN_DIM = 2560;
  const int OUT_DIM = 2560;
  const int LOOP_PER_THREAD = IN_DIM / BlkDimX / 16;
  const int BlkSize = BlkDimX * BlkDimY;
  int in_thread_C_local[M];
  int B_local[1];
  signed char B_decode_local[16];
  signed char A_local[16 * M];
  __shared__ int red_buf0[BlkSize * M];
  #pragma unroll
  for (int i = 0; i < M; i++) {
    in_thread_C_local[i] = 0;
  }
  #pragma unroll
  for (int ax1_0 = 0; ax1_0 < LOOP_PER_THREAD; ++ax1_0) {
    // B_local[0] = *(int*)(B + ((((((int)blockIdx.x) * 3200) + (((int)threadIdx.y) * 640)) + (ax1_0 * 128)) + (((int)threadIdx.x) * 4)));
    B_local[0] = *(int*)(B + ((((((int)blockIdx.x) * blockDim.y * IN_DIM / 4) + (((int)threadIdx.y) * IN_DIM / 4)) + (ax1_0 * IN_DIM / LOOP_PER_THREAD/ 4)) + (((int)threadIdx.x) * 4)));
    decode_i2s_to_i8s_adsbrain(B_local, B_decode_local, 16);
    #pragma unroll
    for (int i = 0; i < M; i++) {
      // *(int4*)(A_local + (i * 16)) = *(int4*)(A + ((ax1_0 * 320) + (((int)threadIdx.x) * 16)) + IN_DIM * i);
      *(int4*)(A_local + (i * 16)) = *(int4*)(A + ((ax1_0 * IN_DIM / LOOP_PER_THREAD) + (((int)threadIdx.x) * 16)) + IN_DIM * i);
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
    ((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] = in_thread_C_local[i];
  }
  __syncthreads();
  if (((int)threadIdx.x) < BlkDimX - 16) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + 16) + i * BlkSize]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 8) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + 8) + i * BlkSize]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 4) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + 4) + i * BlkSize]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 2) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + 2) + i * BlkSize]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 1) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + i * BlkSize] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * blockDim.x) + ((int)threadIdx.x)) + 1) + i * BlkSize]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      D[((((int)blockIdx.x) * blockDim.y) + ((int)threadIdx.y)) + OUT_DIM * i] = (((float)((volatile int*)red_buf0)[(((int)threadIdx.y) * blockDim.x) + i * BlkSize])/__bfloat162float(s[i])*__bfloat162float(ws[0]));
    }
  }
}

template <int M, int BlkDimX, int BlkDimY>
__global__ void __launch_bounds__(128) ladder_int8xint2_1_13824_2560_kernel(int8_t* __restrict__ A, int8_t* __restrict__ B, __nv_bfloat16* __restrict__ D, __nv_bfloat16* __restrict__ s, __nv_bfloat16* __restrict__ ws) {

  const int IN_DIM = 2560;
  const int OUT_DIM = 13824;
  const int LOOP_PER_THREAD = IN_DIM / BlkDimX / 16;
  const int BlkSize = BlkDimX * BlkDimY;
  int in_thread_C_local[M];
  int B_local[1];
  signed char B_decode_local[16];
  signed char A_local[16 * M];
  __shared__ int red_buf0[BlkSize * M];
  #pragma unroll
  for (int i = 0; i < M; i++) {
    in_thread_C_local[i] = 0;
  }
  #pragma unroll
  for (int ax1_0 = 0; ax1_0 < LOOP_PER_THREAD; ++ax1_0) {
    // B_local[0] = *(int*)(B + ((((((int)blockIdx.x) * 3200) + (((int)threadIdx.y) * 640)) + (ax1_0 * 128)) + (((int)threadIdx.x) * 4)));
    B_local[0] = *(int*)(B + ((((((int)blockIdx.x) * blockDim.y * IN_DIM / 4) + (((int)threadIdx.y) * IN_DIM / 4)) + (ax1_0 * IN_DIM / LOOP_PER_THREAD / 4)) + (((int)threadIdx.x) * 4)));
    decode_i2s_to_i8s_adsbrain(B_local, B_decode_local, 16);
    #pragma unroll
    for (int i = 0; i < M; i++) {
      // *(int4*)(A_local + (i * 16)) = *(int4*)(A + ((ax1_0 * 320) + (((int)threadIdx.x) * 16)) + IN_DIM * i);
      *(int4*)(A_local + (i * 16)) = *(int4*)(A + ((ax1_0 * IN_DIM / LOOP_PER_THREAD) + (((int)threadIdx.x) * 16)) + IN_DIM * i);
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
  #pragma unroll
  for (int i = 0; i < M; i++) {
    ((volatile int*)red_buf0)[((((int)threadIdx.y) * BlkDimX) + ((int)threadIdx.x)) + i * BlkSize] = in_thread_C_local[i];
  }
  __syncthreads();
  if (((int)threadIdx.x) < BlkDimX - 16) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * BlkDimX) + ((int)threadIdx.x)) + i * BlkSize] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * BlkDimX) + ((int)threadIdx.x)) + i * BlkSize] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * BlkDimX) + ((int)threadIdx.x)) + 16) + i * BlkSize]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 8) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * BlkDimX) + ((int)threadIdx.x)) + i * BlkSize] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * BlkDimX) + ((int)threadIdx.x)) + i * BlkSize] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * BlkDimX) + ((int)threadIdx.x)) + 8) + i * BlkSize]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 4) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * BlkDimX) + ((int)threadIdx.x)) + i * BlkSize] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * BlkDimX) + ((int)threadIdx.x)) + i * BlkSize] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * BlkDimX) + ((int)threadIdx.x)) + 4) + i * BlkSize]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 2) {
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * BlkDimX) + ((int)threadIdx.x)) + i * BlkSize] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * BlkDimX) + ((int)threadIdx.x)) + i * BlkSize] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * BlkDimX) + ((int)threadIdx.x)) + 2) + i * BlkSize]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 1) {    
    #pragma unroll
    for (int i = 0; i < M; i++) {
      ((volatile int*)red_buf0)[((((int)threadIdx.y) * BlkDimX) + ((int)threadIdx.x)) + i * BlkSize] = (((volatile int*)red_buf0)[((((int)threadIdx.y) * BlkDimX) + ((int)threadIdx.x)) + i * BlkSize] + ((volatile int*)red_buf0)[(((((int)threadIdx.y) * BlkDimX) + ((int)threadIdx.x)) + 1) + i * BlkSize]);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    int out_idx = ((((int)blockIdx.x) * BlkDimY) + ((int)threadIdx.y));
    int ws_idx = out_idx < 6912? 0 : 1;
    #pragma unroll
    for (int i = 0; i < M; i++) {
      D[out_idx + OUT_DIM * i] = (((float)((volatile int*)red_buf0)[(((int)threadIdx.y) * BlkDimX) + i * BlkSize])/__bfloat162float(s[i])*__bfloat162float(ws[ws_idx]));    
    }
  }
}

template <int M, int BlkDimX, int BlkDimY>
__global__ void ladder_int8xint2_1_2560_6912_kernel(int8_t* __restrict__ A, int8_t* __restrict__ B, __nv_bfloat16* __restrict__ D, __nv_bfloat16* __restrict__ s, __nv_bfloat16* __restrict__ ws) {

  const int IN_DIM = 6912;
  const int OUT_DIM = 2560;
  const int LOOP_PER_THREAD = IN_DIM / BlkDimX / 16; 
  const int BlkSize = BlkDimX * BlkDimY;
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
  for (int ax1_0 = 0; ax1_0 < LOOP_PER_THREAD; ++ax1_0) {
    // B_local[0] = *(int*)(B + (((((int)blockIdx.x) * 1728) + (ax1_0 * 432)) + (((int)threadIdx.x) * 4)));
    B_local[0] = *(int*)(B + ((((((int)blockIdx.x) * blockDim.y * IN_DIM / 4) + (((int)threadIdx.y) * IN_DIM / 4)) + (ax1_0 * IN_DIM / LOOP_PER_THREAD / 4)) + (((int)threadIdx.x) * 4)));
    decode_i2s_to_i8s_adsbrain(B_local, B_decode_local, 16);
    #pragma unroll
    for (int i = 0; i < M; i++) {
      // *(int4*)(A_local + (i * 16)) = *(int4*)(A + ((ax1_0 * 1728) + (((int)threadIdx.x) * 16) + IN_DIM * i));
      *(int4*)(A_local + (i * 16)) = *(int4*)(A + ((ax1_0 * IN_DIM / LOOP_PER_THREAD) + (((int)threadIdx.x) * 16)) + IN_DIM * i);
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
    t0_1[i] = __shfl_down_sync(mask_1[0], red_buf0_1[i], 16);
    red_buf0_1[i] = (red_buf0_1[i] + t0_1[i]);
  }
  #pragma unroll
  for (int i = 0; i < M; i++) {
    t0_1[i] = __shfl_down_sync(mask_1[0], red_buf0_1[i], 8);
    red_buf0_1[i] = (red_buf0_1[i] + t0_1[i]);
  }
  #pragma unroll
  for (int i = 0; i < M; i++) {
    t0_1[i] = __shfl_down_sync(mask_1[0], red_buf0_1[i], 4);
    red_buf0_1[i] = (red_buf0_1[i] + t0_1[i]);
  }
  #pragma unroll
  for (int i = 0; i < M; i++) {
    t0_1[i] = __shfl_down_sync(mask_1[0], red_buf0_1[i], 2);
    red_buf0_1[i] = (red_buf0_1[i] + t0_1[i]);
  }
  #pragma unroll
  for (int i = 0; i < M; i++) {
    t0_1[i] = __shfl_down_sync(mask_1[0], red_buf0_1[i], 1);
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
    t0[i] = __shfl_down_sync(mask[0], red_buf0[i], 2);
    red_buf0[i] = (red_buf0[i] + t0[i]);
  }
  #pragma unroll
  for (int i = 0; i < M; i++) {
    t0[i] = __shfl_down_sync(mask[0], red_buf0[i], 1);
    red_buf0[i] = (red_buf0[i] + t0[i]);
  }
  __syncthreads();
  // if (((int)threadIdx.x) == 0) {
  //   // D[((int)blockIdx.x)] = (half)(((float)red_buf0[0])/(float)s[0]*(float)ws[0]);
  //   #pragma unroll
  //   for (int i = 0; i < M; i++) {
  //     D[((int)blockIdx.x) + OUT_DIM * i] = (half)(((float)red_buf0[i])/__bfloat162float(s[i])*__bfloat162float(ws[0]));  //((half) ((float)red_buf0[i]));  //  
  //   }
  // }

  if (((int)threadIdx.x) == 0) {
    int out_idx = ((((int)blockIdx.x) * BlkDimY) + ((int)threadIdx.y));
    int ws_idx = 0;
    #pragma unroll
    for (int i = 0; i < M; i++) {
      D[out_idx + OUT_DIM * i] = (((float)((volatile int*)red_buf0)[(((int)threadIdx.y) * BlkDimX) + i * BlkSize])/__bfloat162float(s[i])*__bfloat162float(ws[ws_idx]));    
    }
  }
}


