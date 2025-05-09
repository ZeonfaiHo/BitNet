#include "bitnet_kernels.h"
#include "bitnet_kernels_d2560.h"

// #include <torch/extension.h>


// torch::Tensor bitlinear_int8xint2(torch::Tensor act, torch::Tensor weight, torch::Tensor act_s, torch::Tensor weight_s)
// {
//     int M = act.sizes()[0];
//     int N = weight.sizes()[0];
//     int K = weight.sizes()[1] * 4;
//     torch::at::cuda::CUDAStream stream = torch::at::cuda::getCurrentCUDAStream();
//     auto ret = torch::zeros({M, N}, torch::kHalf);

//     int8_t* input0 = act.data<int8_t>();
//     int8_t* input1 = weight.data<int8_t>();
//     half* output0 = ret.data<half>();
//     half* s = act_s.data<half>();
//     half* ws = weight_s.data<half>();

//     if (M == 1 && N == 4800 && K == 3200)
//     {
//         ladder_int8xint2_1_4800_3200_kernel<<<dim3(300, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
//     }
//     else if (M == 1 && N == 3200 && K == 3200)
//     {
//         ladder_int8xint2_1_3200_3200_kernel<<<dim3(200, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
//     }
//     else if (M == 1 && N == 20480 && K == 3200)
//     {
//         ladder_int8xint2_1_20480_3200_kernel<<<dim3(640, 1, 1), dim3(4, 32, 1), 0, stream>>>(input0, input1, output0, s, ws);
//     }
//     else if (M == 1 && N == 3200 && K == 10240)
//     {
//         ladder_int8xint2_1_3200_10240_kernel<<<dim3(200, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
//     }
//     else
//     {
//         std::cout << "required ladder gemm kernel: M " << M << ", N " << N << ", K " << K << std::endl;
//     }

//     return ret;
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("bitlinear_int8xint2", &bitlinear_int8xint2, "bitlinear_int8xint2 forward (CUDA)");
//   }

extern "C" void bitlinear_int8xint2(int8_t* input0, int8_t* input1, __nv_bfloat16* output0, __nv_bfloat16* s, __nv_bfloat16* ws, int M, int N, int K, cudaStream_t stream)
{
    if (M == 1) {
        // 1.5b model, hidden_dim=3200
        if (N == 4800 && K == 3200)
        {
            ladder_int8xint2_1_4800_3200_kernel<1><<<dim3(960, 1, 1), dim3(25, 5, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        else if (N == 3200 && K == 3200)
        {
            ladder_int8xint2_1_3200_3200_kernel<1><<<dim3(640, 1, 1), dim3(25, 5, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        else if (N == 20480 && K == 3200)
        {
            ladder_int8xint2_1_20480_3200_kernel<1><<<dim3(4096, 1, 1), dim3(25, 5, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        else if (N == 3200 && K == 10240)
        {
            ladder_int8xint2_1_3200_10240_kernel<1><<<dim3(3200, 1, 1), dim3(128, 1, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        // 2b model, hidden_dim=2560
        else if (N == 3840 && K == 2560)
        {
            ladder_int8xint2_1_3840_2560_kernel<1, 32, 4><<<dim3(960, 1, 1), dim3(32, 4, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        else if (N == 2560 && K == 2560)
        {
            ladder_int8xint2_1_2560_2560_kernel<1, 32, 4><<<dim3(640, 1, 1), dim3(32, 4, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        else if (N == 13824 && K == 2560)
        {
            ladder_int8xint2_1_13824_2560_kernel<1, 32, 4><<<dim3(3456, 1, 1), dim3(32, 4, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        else if (N == 2560 && K == 6912)
        {
            ladder_int8xint2_1_2560_6912_kernel<1, 108, 1><<<dim3(2560, 1, 1), dim3(108, 1, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        else
        {
            std::cout << "required ladder gemm kernel: M " << M << ", N " << N << ", K " << K << std::endl;
        }
    }
    if (M == 2) {
        if (N == 4800 && K == 3200)
        {
            ladder_int8xint2_1_4800_3200_kernel<2><<<dim3(960, 1, 1), dim3(25, 5, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        else if (N == 3200 && K == 3200)
        {
            ladder_int8xint2_1_3200_3200_kernel<2><<<dim3(640, 1, 1), dim3(25, 5, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        else if (N == 20480 && K == 3200)
        {
            ladder_int8xint2_1_20480_3200_kernel<2><<<dim3(4096, 1, 1), dim3(25, 5, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        else if (N == 3200 && K == 10240)
        {
            ladder_int8xint2_1_3200_10240_kernel<2><<<dim3(3200, 1, 1), dim3(128, 1, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        else
        {
            std::cout << "required ladder gemm kernel: M " << M << ", N " << N << ", K " << K << std::endl;
        }
    }

    if (M == 3) {
        if (N == 4800 && K == 3200)
        {
            ladder_int8xint2_1_4800_3200_kernel<3><<<dim3(960, 1, 1), dim3(25, 5, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        else if (N == 3200 && K == 3200)
        {
            ladder_int8xint2_1_3200_3200_kernel<3><<<dim3(640, 1, 1), dim3(25, 5, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        else if (N == 20480 && K == 3200)
        {
            ladder_int8xint2_1_20480_3200_kernel<3><<<dim3(4096, 1, 1), dim3(25, 5, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        else if (N == 3200 && K == 10240)
        {
            ladder_int8xint2_1_3200_10240_kernel<3><<<dim3(3200, 1, 1), dim3(128, 1, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        else
        {
            std::cout << "required ladder gemm kernel: M " << M << ", N " << N << ", K " << K << std::endl;
        }
    }

    if (M == 4) {
        if (N == 4800 && K == 3200)
        {
            ladder_int8xint2_1_4800_3200_kernel<4><<<dim3(960, 1, 1), dim3(25, 5, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        else if (N == 3200 && K == 3200)
        {
            ladder_int8xint2_1_3200_3200_kernel<4><<<dim3(640, 1, 1), dim3(25, 5, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        else if (N == 20480 && K == 3200)
        {
            ladder_int8xint2_1_20480_3200_kernel<4><<<dim3(4096, 1, 1), dim3(25, 5, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        else if (N == 3200 && K == 10240)
        {
            ladder_int8xint2_1_3200_10240_kernel<4><<<dim3(3200, 1, 1), dim3(128, 1, 1), 0, stream>>>(input0, input1, output0, s, ws);
        }
        else
        {
            std::cout << "required ladder gemm kernel: M " << M << ", N " << N << ", K " << K << std::endl;
        }
    }

    if (M > 4) {
        std::cout << "required ladder gemm kernel: M " << M << ", N " << N << ", K " << K << std::endl;
    }
}