nvcc -std=c++17 -Xcudafe --diag_suppress=177 --compiler-options -fPIC -lineinfo --shared bitnet_kernels.cu -lcuda -gencode=arch=compute_80,code=compute_80 -I/home/lingm/projects/bitnet/ladder/ladder_cutlass/include -o libbitnet.so


