/*
 * Net.h
 *
 *  Created on: Dec 12, 2017
 *      Author: hee
 */

//initialization change
//implementation of  each version
//print reslut of each version

#ifndef NET_H_HEE
#define NET_H_HEE

#include <iostream>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <time.h>
#include <unistd.h>

//weight reading
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <string>

//weight reading end

using namespace std;
using namespace thrust;

void clear();
void printMNIST_HW_row_col_for_main(device_vector<float>& data_tmp,int height, int width, int row_start_index, int col_start_index,
		int row_num, int col_num, int row_interval, int col_interval, char* str);

//minibatch 2: 2e-4 -> 25 epoch 97%
//minibatch 2: 2e-4 -> 25 epoch 97%
//minibatch 2: 1e-4 -> 90 epoch 97.81%
//minibatch 2: 1e-5 -> 785 epoch 98.02%

#define TILE_WIDTH 32
#define MINIBATCH 1000
#define LEARNIG_RATE 2e-4
#define LAMDA (3e-1)
#define CONV_KERNEL_SIZE 13
#define LABEL_ONE 1
#define LABEL_ZERO 0
#define NUM_TRAIN 60000
#define NUM_TEST  10000
#define RAW_DIM   28
#define RAW_PIXELS_PER_IMG 784 			// 28x28, single channel image
#define RAW_DIM_PADDING 32
#define RAW_PIXELS_PER_IMG_PADDING 1024
#define MNIST_SCALE_FACTOR 0.00390625	// 1/255
#define MAXBYTE 255


/***** Function declarations ***************************/
void printMNIST(host_vector<float>& data, int after_minib);
void printMNIST_num(host_vector<float>& data, int label, int num);
void printMNIST_HW_row_col(device_vector<float>& data_tmp,int height, int width, int row_start_index, int col_start_index,
		int row_num, int col_num, int row_interval, int col_interval, char* str);
void printMNIST_HW_avg_value(device_vector<float>& data_tmp, char* str);
void printMNIST_HW_avg_value(host_vector<float>& data, char* str);
void read_data(const char* datapath, host_vector< host_vector<float> >& data);
void read_data_no_padding(const char* datapath, host_vector< host_vector<float> >& data);
void read_label(const char* labelPath, host_vector<int>& label);
void flatten(host_vector< host_vector<float> >& input, host_vector<float>& output);
//size_in -> entire width of vector(MINIBATCH*Outputimage_channel*Outputimage_height*Outputimage_width)
void forward_relu(device_vector<float>& input, device_vector<float>& output);
//size_in -> entire width of vector(MINIBATCH*Outputimage_channel*Outputimage_height*Outputimage_width)
void backward_relu(device_vector<float>& input, device_vector<float>& output);
void reduceTofirstindex(device_vector<float>& input, int H_in, int W_in);
void reduceTofirstindex(float* input_pointer, int H_in, int W_in);
//option 1 -> forward
//option 2 -> backward
void relu_h_gpu_test(host_vector<float>& input, device_vector<float>& comp, int size_in, int test_number, int option);
void sigmoid(device_vector<float>& input, int size_in);
//size_in -> entire width of vector(MINIBATCH*Outputimage_channel*Outputimage_height*Outputimage_width)
void backward_sigmoid(device_vector<float>& input, int size_in);
void forward_bias_per_channel(device_vector<float>& input, device_vector<float>& bias, int N, int ch_in, int h_in, int w_in);
void backward_bias_per_channel(device_vector<float>& input, device_vector<float>& bias, int N, int h_in, int w_total_in,
		int w_ch, int w_width_mul_w_height);
void backward_bias(device_vector<float>& input, device_vector<float>& bias, int N, int ch_in, int h_in, int w_in);
void forward_bias_gpu_test(host_vector<float>& input, device_vector<float>& bias, device_vector<float>& comp,
		int N, int ch_in, int h_in, int w_in, int test_number);
void forward_sigmoid_gpu_test(host_vector<float>& input, device_vector<float>& comp, int size_in, int test_number);
void transposeMatrix(device_vector<float>& XT, device_vector<float>& X, int X_height, int X_width);
void transposeMatrix(float* XT_pointer, float* X_pointer, int input_height, int input_width);
void transposeMatrix_gpu_test(host_vector<float>& Output_c, host_vector<float>& input_c, int height_in, int width_in, int test_number);
//size_in -> entire width of vector(MINIBATCH*Outputimage_channel*Outputimage_height*Outputimage_width)
void div_by_constant(device_vector<float>& input, int n, int size_in);
//blocknumber -> Input_width/1024
__global__ void div_by_constant(float* X, int n, int size_in);
__global__ void forward_bias(float* X, float* b, int N, int ch_in, int h_in, int w_in);
__global__ void sigmoid(float* X, int size_in);
//blocknumber -> size_in/1024
__global__ void backward_sigmoid(float* X, int size_in);
//bx = output_WIDTH, by = output_HEIGH
__global__ void gemm_h(float* Md, float* Nd, float* Pd, int M_height_in, int M_width_N_height_in, int N_width_in, int height_out, int width_out);
__global__ void gemm_with_bias_h(float* Md, float* Nd, float* Pd, float* B, int M_height_in, int M_width_N_height_in, int N_width_in , int height_out, int width_out);
//bx = input_WIDTH, by = input_HEIGHT
__global__ void transposeMatrix_h(float *odata, const float *idata, int height_in, int width_in);
//bx*tx = idata_width*idata*height
__global__ void grad_descent(float *odata, const float *idata, int size);
//blocknumber -> size_in/1024
__global__ void init_zero_vector(float *X, int size_in);
//blocknumber -> size_in/1024
__global__ void relu_h(float* X, float* Y, int size_in);
//blocknumber -> size_in/1024
__global__ void backward_relu_h(float* X, float* Y, int size_in);
//blocknumber -> Input_width/1024
__global__ void reduceTofirstindex_h(float* X, int H_in, int W_in);
__global__ void backward_bias(float* X, float* b, int N, int ch_in, int h_in, int w_in);

//For thrust calculation
struct square { __host__ __device__ float operator()(float x) { return x * x; } };
struct div_h { __host__ __device__ float operator()(float x) { return x / MINIBATCH; } };
struct mul_h { __host__ __device__ float operator()(float x) { return x * MINIBATCH; } };
struct plus_h {
	const float weight_decay_h;

	__host__ __device__ plus_h(const float weight_decay):weight_decay_h(weight_decay){}
	__host__ __device__ float operator()(float x) {
	return x + LAMDA*weight_decay_h; }
};





/***** CLASS declarations ***************************/

class FullyConnect{

public:
	void init(int X_h,
			int X_w_W_h,
			int W_w);
	void forward(device_vector<float>& input);
	void forward();
	void forward_gpu_test(device_vector<float>& input, int test_number);
	void backward();



	// M*(C*H*W)
	host_vector<float> X_c; // when back, dE_dX
	host_vector<float> W_c;
    host_vector<float> b_c;
	host_vector<float> Wgrad_c;
    host_vector<float> bgrad_c;
	host_vector<float> Output_c;
	device_vector<float> X; // when back, dE_dX
	device_vector<float> XT;
	device_vector<float> W;
	device_vector<float> WT;
	device_vector<float> b;
	device_vector<float> Wgrad;
	device_vector<float> bgrad;
	device_vector<float> Output;// when back, dE_dY
	device_vector<float> OutputT;

	int X_width;
	int X_height;
	int XT_width;
	int XT_height;
	int W_width;
	int W_height;
	int WT_width;
	int WT_height;
	int Output_width;
	int Output_height;
	int OutputT_width;
	int OutputT_height;
};











class Convolution{ // M*C*H*W
public:
	void init(int minib, int X_h, int X_w,int X_ch, int W_w_h, int W_ch);
	void forward_CPU();
	void forward_GPU_gemm();
	void forward_GPU_naive();
	void forward_GPU_tiled();
	void backward_GPU_gemm();
	void forward_cpu_test(host_vector<float>& input, int test_number);
	void forward_gpu_test(device_vector<float>& input, int test_number);
	void backward_col2im_gpu_test(int test_number);
	void backward();
	void backward_GPU_naive();

	void convLayer_forward(int N, float* X, int C, int H_in, int W_in, float* W, int K, float* Y, int M);
	void convLayer_backward_xgrad(int N, int M, int C, int H_in,
			int W_in, int K, float* dE_dY, float* W, float* dE_dX);
	void convLayer_backward_wgrad(int N, int M, int C, int H_in, int W_in,
			int K, float* dE_dY, float* X,
			float* dE_dW);

	host_vector<float> X_c; // when back, dE_dX
	host_vector<float> W_c;
    host_vector<float> b_c;
	host_vector<float> Wgrad_c;
	host_vector<float> Output_c;
    host_vector<float> bgrad_c;
    host_vector<float> Unroll_X_c;
	device_vector<float> X; // when back, dE_dX
	device_vector<float> W;
	device_vector<float> WT;
	device_vector<float> b;
	device_vector<float> Wgrad;
	device_vector<float> WgradTmp;
	device_vector<float> Output; // when back, dE_dY
	device_vector<float> bgrad;
	device_vector<float> Unroll_X;
	device_vector<float> Unroll_XT;



	int W_width_height;
	int W_channel;
	int X_width;
	int X_height;
	int Unroll_X_width;
	int Unroll_X_height;
	int Inputimage_width;
	int Inputimage_height;
	int Inputimage_channel;
	int minibatch;
	int Outputimage_width;
	int Outputimage_height;
	int Outputimage_channel;
	int Output_width;
	int Output_height;
};

__global__ void convLayer_forward_GPU_naive(float* X, float* W, float* Y,
		int C, int H_in, int W_in, int W_out, int K, int M);
__global__ void convLayer_forward_GPU_tiled(float* X, float* W, float* Y,
		int C, int H_in, int W_in, int W_out, int K, int M);
__global__ void convLayer_backward_GPU_naive(float* X, float* W, float* Y,
		int C, int H_in, int W_in, int W_out, int K, int M);
__global__ void unroll_Kernel(int C, int H_in, int W_in, int K, float* X, float* X_unroll);
__global__ void col2im_Kernel(int C, int H_in, int W_in, int K, float* X, float* X_unroll);









class Pool{ // M*C*H*W
public:
	void init(int minib, int X_h, int X_w, int X_ch, int pool_size);
	void poolingLayer_forward(int N, float* X, int H_in, int W_in, float* Y, int M);

	void forward_CPU(host_vector<float>& input);
	void forward_GPU_naive(device_vector<float> & input);
	void forward_GPU_tiled(device_vector<float>& input);
	void backward_GPU(device_vector<float> & output);

	host_vector<float> X_c;
	device_vector<float> X;
	host_vector<float> Output_c;
	device_vector<float> Output;
	device_vector<float> b;
	device_vector<float> b_c;

	int X_height;
	int X_width;
	int b_height;
	int b_width;
	int Inputimage_height;
	int Inputimage_width;
	int Inputimage_channel;
	int Outputimage_height;
	int Outputimage_width;
	int Outputimage_channel;
	int Output_height;
	int Output_width;
	int minibatch;
	int pool_size;
};
__global__ void poolingLayer_forward_GPU_naive(float* X, int H_in, int W_in, float* Y, int M, int pool_size);
__global__ void poolingLayer_forward_GPU_tiled(float* X, int H_in, int W_in, float* Y, int M, int pool_size);
__global__ void poolingLayer_backward_GPU(float* X, int H_in, int W_in, float* Y, int M, int pool_size);







class Softmax{ // M*C*H*W
public:
	host_vector<float> delta_c;
	device_vector<float> delta;
	float loss;

	void cross_entropy_loss(int N, host_vector<int> label, host_vector<float>& input, int Width_in, float& loss, int minib);
	void softmax_backward(int N, host_vector<int> label, host_vector<float>& softmax_output,
			host_vector<float>& delta, int Width_in, int minib);
	void softmax(int N, int Width_in,
			host_vector<int>& label, host_vector<float>& output);
	void accuracy(int N, int Width_in, host_vector< host_vector<float> >& Xtrain,
			host_vector<int>& label, host_vector<float>& output, int minib, int & correct_num);


};


class Net
{
public:
	virtual void train(host_vector< host_vector<float> >& Xtrain, host_vector<int>& Ytrain)=0;
	virtual void test(host_vector< host_vector<float> >& Xtest, host_vector<int>& Ytest)=0;

	Convolution conv1, conv2;
	Pool pool1, pool2;
	FullyConnect fc1, fc2, fc3;
	Softmax sm1;
	int correct_num;
};


class Net_CPU : public Net
{
public:
	Net_CPU();
	~Net_CPU();
	virtual void train(host_vector< host_vector<float> >& Xtrain, host_vector<int>& Ytrain) override;
	virtual void test(host_vector< host_vector<float> >& Xtest, host_vector<int>& Ytest) override;

};

class Net_GPU_Naive : public Net
{
public:
	Net_GPU_Naive();
	~Net_GPU_Naive();
	virtual void train(host_vector< host_vector<float> >& Xtrain, host_vector<int>& Ytrain) override;
	virtual void test(host_vector< host_vector<float> >& Xtest, host_vector<int>& Ytest) override;
};




class Net_GPU_Tiled : public Net
{
public:
	Net_GPU_Tiled();
	~Net_GPU_Tiled();
	virtual void train(host_vector< host_vector<float> >& Xtrain, host_vector<int>& Ytrain) override;
	virtual void test(host_vector< host_vector<float> >& Xtest, host_vector<int>& Ytest) override;
};



class Net_GPU_Gemm : public Net
{
public:
	Net_GPU_Gemm();
	~Net_GPU_Gemm();
	virtual void train(host_vector< host_vector<float> >& Xtrain, host_vector<int>& Ytrain) override;
	virtual void test(host_vector< host_vector<float> >& Xtest, host_vector<int>& Ytest) override;
};


class Net_GPU_test : public Net
{
public:
	Net_GPU_test();
	~Net_GPU_test();
	virtual void train(host_vector< host_vector<float> >& Xtrain, host_vector<int>& Ytrain) override;
	virtual void test(host_vector< host_vector<float> >& Xtest, host_vector<int>& Ytest) override;
	void simpletest();
};



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#endif
