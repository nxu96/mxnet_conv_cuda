#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

// TILE_SIZE 
#define TILE_SIZE_1 16
#define TILE_SIZE_2 24
#include <mxnet/base.h>
#include <iostream>
namespace mxnet
{
namespace op
{
// forward kernel
__global__ void forward_kernel_op1(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, const int H_out, const int W_out)
{

    /////////Ning's implementation////////////
    // unroll + shared memory tiled matrix multiply
    int b = blockIdx.z; // Batch index
    int tx = threadIdx.x; // thread index x
    int ty = threadIdx.y; // thread index y
    int xpos = blockIdx.x * TILE_SIZE_1 + tx; 

    // starting position of x and y
    x += b * H * W; // There is no C in op1 
    y += b * M * H_out * W_out; //current batch (M * H_out * W_out)
    // SM size = block size 
    __shared__ float tileK[TILE_SIZE_1][TILE_SIZE_1];
    __shared__ float tileX[TILE_SIZE_1][TILE_SIZE_1];

    // Unroll matrix X + tiled matrix multiplication using shared memory
    // unrolled are K and X matrix 
    int X_row = K * K; // K*K = 49 
    int K_col = X_row; 
    //int K_row = M; 
    // X_row = K_col
    //int X_col = H_out * W_out; 
    int h_unroll, w_unroll, q, p;
    float ans = 0; 

    for (int tile_idx = 0; tile_idx < 4; tile_idx++){
        // load filter into shared memory 
        int K_col_idx = tile_idx * TILE_SIZE_1 + tx;
        if (K_col_idx < K_col && ty < M){ 
            tileK[ty][tx] = k[ty * X_row + K_col_idx]; // k: M*K*K
        }
        else{
            tileK[ty][tx] = 0;
        }

        // load input x into shared memory
        int X_row_idx = tile_idx * TILE_SIZE_1 + ty;
        // within the range of X matrix 
        if (X_row_idx < X_row && xpos < H_out * W_out){ 
            w_unroll = xpos % W_out;
            h_unroll = xpos / W_out;
            q = X_row_idx % K;
            X_row_idx /= K; 
            p = X_row_idx % K; 
            tileX[ty][tx] = x[(p + h_unroll) * W  + w_unroll + q];
        }
        else{
            tileX[ty][tx] = 0;
        }
        __syncthreads();

        for (int n = 0; n < TILE_SIZE_1; ++n){
            ans += tileK[ty][n] * tileX[n][tx];
        }

        __syncthreads();
    }

    if ( xpos < H_out * W_out && ty < M ){
        y[ty * H_out * W_out + xpos] = ans;
    }
    ////////Ning's implementation ends here///////
    
}

__global__ void forward_kernel_op2(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, const int H_out, const int W_out){

    /////////Ning's implementation////////////
    // unroll + shared memory tiled matrix multiply
    int b = blockIdx.z; // Batch index
    int tx = threadIdx.x; // thread index x
    int ty = threadIdx.y; // thread index y
    int xpos = blockIdx.x * TILE_SIZE_2 + tx; 

    // starting position of x and y
    x += b * C * H * W; //  
    y += b * M * H_out * W_out; //current batch (M * H_out * W_out)
    // SM size = block size 
    __shared__ float tileK[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float tileX[TILE_SIZE_2][TILE_SIZE_2];

    // Unroll matrix X + tiled matrix multiplication using shared memory
    // unrolled are K and X matrix 
    int X_row = C * K * K;  
    int K_col = X_row; 
    int h_unroll, w_unroll, q, p, c;
    float ans = 0; 
    // 7 * 7 * 12 = 588 < 24 * 25 = 600;
    for (int tile_idx = 0; tile_idx < 25; tile_idx++){
        // load filter into shared memory 
        int K_col_idx = tile_idx * TILE_SIZE_2 + tx;
        if (K_col_idx < K_col){ 
            tileK[ty][tx] = k[ty * X_row + K_col_idx]; // k: M*K*K
        }
        else{
            tileK[ty][tx] = 0;
        }

        // load input x into shared memory
        int X_row_idx = tile_idx * TILE_SIZE_2 + ty;
        // within the range of X matrix 
        if (X_row_idx < X_row){ 
            w_unroll = xpos % W_out;
            h_unroll = xpos / W_out;
            q = X_row_idx % K;
            X_row_idx /= K; 
            p = X_row_idx % K;
            c = X_row_idx / K; 
            tileX[ty][tx] = x[(p + h_unroll) * W + c * H * W + w_unroll + q];
        }
        else{
            tileX[ty][tx] = 0;
        }
        __syncthreads();

        for (int n = 0; n < TILE_SIZE_2; ++n){
            ans += tileK[ty][n] * tileX[n][tx];
        }

        __syncthreads();
    }

    if ( xpos < H_out * W_out){
        y[ty * H_out * W_out + xpos] = ans;
    }
    ////////Ning's implementation ends here///////

}


//TODO: This is the host code
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // // Use mxnet's CHECK_EQ to do assertions.
    // CHECK_EQ(0, 1) 
    
    const int B = x.shape_[0]; // Batch
    const int M = y.shape_[1]; // num_filter, M channels
    const int C = x.shape_[1]; // Input channles 
    const int H = x.shape_[2]; // Input Height
    const int W = x.shape_[3]; // Input Width
    const int K = w.shape_[3]; // filter size K * K 
    const int H_out = H - K + 1; // output Height
    const int W_out = W - K + 1; // output Width
    // std::cout << "Batch Size is: " << B << std::endl;
    // std::cout << "Output Channel is: " << M << std::endl;
    // std::cout << "Output Channel is: " << C << std::endl;
    // std::cout << "Input Height is: " << H << std::endl;
    // std::cout << "Input Width is: " << W << std::endl;
    // std::cout << "Filter Size is: " << K << std::endl;

    ///////////////////////////
    // TODO: OP2 version TBD
    if (C > 1){ 
        // Op2: B = 10000 M = 24 C = 12 H = 33 W = 33 K = 7 
        // forward_kernel_op2
        int gridy = ceil(1.0 * M / TILE_SIZE_2);
        int gridx = ceil(1.0 * H_out * W_out / TILE_SIZE_2);
        dim3 gridDim(gridx, gridy, B); 
        dim3 blockDim(TILE_SIZE_2, TILE_SIZE_2, 1); 
        MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
        forward_kernel_op2<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K, H_out, W_out);
        MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    }
    else{ 
        // Op1: B = 10000 M = 12 C = 1 H = 72 W = 72 K = 7 
        // call the kernel here 
        int gridy = ceil(1.0 * M / TILE_SIZE_1);
        int gridx = ceil(1.0 * H_out * W_out / TILE_SIZE_1);
        // M * H * W * B output
        dim3 gridDim(gridx, gridy, B); 
        dim3 blockDim(TILE_SIZE_1, TILE_SIZE_1, 1); 
        MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
        forward_kernel_op1<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K, H_out, W_out);
        MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    }
    
    ///////////////////////////

}



///////////////////////////////////////////////////////////////////////////////////////////
/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    assert(0 && "No forward implementation for other datatypes needed");
}
}
}

#endif