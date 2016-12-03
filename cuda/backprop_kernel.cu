#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "backprop.h"

#define TILE_SIZE 16

////////////////////////////////////////////////////////////////////////////////

extern void bpnn_layerforward(float *l1, float *l2, float *conn, int n1, int n2);

extern void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float *who, float *hidden, float *err);

extern void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float *w, float *oldw);


extern int setup(int argc, char** argv);

extern float *alloc_2d_dbl(int m, int n);

extern float squash(float x);

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
  setup(argc, argv);
}


void bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
   
  printf("Performing CPU computation\n");
  bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);
  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

}


__global__ void bpnn_layerforward(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    __shared__ float ds_M[TILE_SIZE][TILE_SIZE];
    __shared__ float ds_N[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;

    for (int i = 0 ; i < (k - 1)/TILE_SIZE+1 ; i++) {
      if(row < m && (i * TILE_SIZE + threadIdx.x) < k) {
        ds_M[threadIdx.y][threadIdx.x] = A[row * k + i*TILE_SIZE + threadIdx.x];
      } else {
        ds_M[threadIdx.y][threadIdx.x] = 0;
      }
      if(col < n && i*TILE_SIZE + threadIdx.y < k) {
        ds_N[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * n + col];
      } else {
        ds_N[threadIdx.y][threadIdx.x] = 0;
      }

      __syncthreads();
      if (row < m && col < n) {
        for(int j = 0 ; j < TILE_SIZE ; j++) {
          sum += ds_M[threadIdx.y][j] * ds_N[j][threadIdx.x];
        }
      }
      __syncthreads();
    } 

    if(row < m && col < n) { 
      C[row * n + col] = sum;
    }

}