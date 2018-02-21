#include "Net_h.h"


void Pool::init(int minib, int Inputimage_h, int Inputimage_w, int Inputimage_ch, int pool_size)
{
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0,1.0);

	this->Inputimage_height=Inputimage_h;
	this->Inputimage_width=Inputimage_w;
	this->Inputimage_channel=Inputimage_ch;
	this->Outputimage_height=Inputimage_h/pool_size;
	this->Outputimage_width=Inputimage_w/pool_size;
	this->Outputimage_channel=Inputimage_ch;
	this->Output_height=minib;
	this->Output_width=Outputimage_channel*Outputimage_height*Outputimage_width;
	this->minibatch=minib;
	this->X_height=minib;
	this->X_width=Inputimage_channel*Inputimage_height*Inputimage_width;
	this->b_height=minib;
	this->b_width=Inputimage_channel;
	this->pool_size=pool_size;

	this->X_c.resize(minibatch*Inputimage_channel*Inputimage_height*Inputimage_width, 0);
	this->X.resize(minibatch*Inputimage_channel*Inputimage_height*Inputimage_width, 0);
	this->Output_c.resize(minibatch*Outputimage_channel*Outputimage_height*Outputimage_width, 0);
	this->Output.resize(minibatch*Outputimage_channel*Outputimage_height*Outputimage_width, 0);
	this->b.resize(Inputimage_channel,0.1);
	this->b_c.resize(Inputimage_channel,0.1);
}

void Pool::forward_CPU(host_vector<float> & input)
{
	float* input_pointer = thrust::raw_pointer_cast( input.data() );
	float* Output_pointer = thrust::raw_pointer_cast( Output_c.data() );
	poolingLayer_forward(minibatch, input_pointer, Inputimage_height, Inputimage_width, Output_pointer, Outputimage_channel);
}

void Pool::forward_GPU_naive(device_vector<float> & input)
{
	dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH);
	int bz = ceil((float)Outputimage_width/TILE_WIDTH)*ceil((float)Outputimage_height/TILE_WIDTH);
	if( bz == 0 )
		bz = 1;
	dim3 numBlocks(minibatch, Outputimage_channel, bz);

	float* input_pointer = thrust::raw_pointer_cast( input.data() );
	float* Output_pointer = thrust::raw_pointer_cast( Output.data() );
	poolingLayer_forward_GPU_naive<<<numBlocks,threadsPerBlock>>>(input_pointer, Inputimage_height,
			Inputimage_width, Output_pointer, Outputimage_channel, pool_size);

}

//double for loop version
void Pool::backward_GPU(device_vector<float> & output)
{
	dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH);
	int bz = ceil((float)Outputimage_width/TILE_WIDTH)*ceil((float)Outputimage_height/TILE_WIDTH);
	if( bz == 0 )
		bz = 1;
	dim3 numBlocks(minibatch, Outputimage_channel, bz);

	float* input_pointer = thrust::raw_pointer_cast( X.data() );
	float* output_pointer = thrust::raw_pointer_cast( output.data() );

	poolingLayer_backward_GPU<<<numBlocks,threadsPerBlock>>>(input_pointer, Inputimage_height,
			Inputimage_width, output_pointer, Outputimage_channel, pool_size);
}

void Pool::forward_GPU_tiled(device_vector<float>& input)
{
	dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH);
	//if TILE WIDTH is bigger than output feature maps's width,
	//boundary check is needed
	int bz = ceil((float)Outputimage_width/TILE_WIDTH)*ceil((float)Outputimage_height/TILE_WIDTH);
	if( bz == 0 )
		bz = 1;
	dim3 numBlocks(minibatch, Outputimage_channel , bz);
	//bx = minibatch, by = Outputimage_channel bz = number of TILES in output feature map

	float* input_pointer = thrust::raw_pointer_cast( input.data() );
	float* Output_pointer = thrust::raw_pointer_cast( Output.data() );

	poolingLayer_forward_GPU_tiled<<<numBlocks,threadsPerBlock>>>(input_pointer, Inputimage_height,
			Inputimage_width, Output_pointer, Outputimage_channel, pool_size);
}


//M: number of input, output feature maps
//H_in: height of each input image
//W_in: width of each input map image
//X: input feature maps
//Y: output feature maps
void Pool::poolingLayer_forward(int N, float* X, int H_in, int W_in, float* Y, int M)
{
	int n, m, h, w, p, q;
	int H_out = H_in/pool_size;
	int W_out = W_in/pool_size;
	for (n = 0; n < N; n++) // for each sample in the mini-batch
		for (m = 0; m < M; m++) // for each output featrue maps
			for (h = 0; h < H_in / pool_size; h++) // for each output element
				for (w = 0; w < W_in / pool_size; w++) {
					Y[n*(M*H_out*W_out)+ m*(H_out*W_out) + h*(W_out) + w] = 0;
					for (p = 0; p < pool_size; p++) { // loop over KxK input samples
						for (q = 0; q < pool_size; q++)
							Y[n*(M*H_out*W_out)+ m*(H_out*W_out) + h*(W_out) + w] =
									Y[n*(M*H_out*W_out)+ m*(H_out*W_out) + h*(W_out) + w] +
									X[n*(M*H_in*W_in)+ m*(H_in*W_in) +
									  (pool_size * h + p)*(W_in) + (pool_size * w + q)] / (pool_size * pool_size);
					}
#ifdef DEBUG_POOLING_
					if(n==7 && m == 5 && w==13) printf("h=%d, w=%d, acc=%.3lf\n", h, w, Y[n*(M*H_out*W_out)+ m*(H_out*W_out) + h*(W_out) + w]);
#endif
				}
}

//M: number of input, output feature maps
//H_in: height of each input image
//W_in: width of each input map image
//X: input feature maps
//Y: output feature maps
__global__ void poolingLayer_forward_GPU_naive(float* X, int H_in, int W_in, float* Y, int M, int pool_size)
{
	int n, m, h, w, p, q;
	int H_out = H_in/pool_size;
	int W_out = W_in/pool_size;
	int W_grid = ceilf((float)W_out/TILE_WIDTH);
	if(W_grid==0)
		W_grid = 1;
	n = blockIdx.x;
	m = blockIdx.y;
	h = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y;
	w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;
	//h and w is not center point of calculating, it's upper left corner point of Input image
	float acc = 0;
	for (p = 0; p < pool_size; p++) { // loop over KxK input samples
		for (q = 0; q < pool_size; q++)
			if(h < H_out && w < W_out)
				acc = acc + X[n*(M*H_in*W_in)+ m*(H_in*W_in) +
				              (pool_size * h + p)*(W_in) + (pool_size * w + q)] / (pool_size * pool_size);
	}
	__syncthreads();
	if(h < H_out && w < W_out)
	{
		Y[n*(M*H_out*W_out)+ m*(H_out*W_out) + h*(W_out) + w] = acc;
#ifdef DEBUG_POOLING_
		if(n==7 && m == 5 && w==13) printf("h=%d, w=%d, acc=%.3lf\n", h, w, acc);
#endif
	}
}

//M: number of input, output feature maps
//H_in: height of each input image
//W_in: width of each input map image
//X: input feature maps
//Y: output feature maps
__global__ void poolingLayer_forward_GPU_tiled(float* X, int H_in, int W_in, float* Y, int M, int pool_size)
{
	int n, m, h0, w0, h_base, w_base, h, w;
	int H_out = H_in/pool_size;
	int W_out = W_in/pool_size;
	int X_tile_width = TILE_WIDTH;
	int W_grid = ceilf((float)W_out/TILE_WIDTH);
	if(W_grid==0)
		W_grid = 1;

	n = blockIdx.x;
	m = blockIdx.y;

	//extern __shared__ float shmem[]; // dynamic run time allocation , not work

	__shared__ float shmem[(TILE_WIDTH)*(TILE_WIDTH)];

	float* X_shared = (float*)&shmem[0];//allocation index 0

	n = blockIdx.x;
	m = blockIdx.y;

	// vertical base out data index for the block
	//blockIdx.z -> number of TILES needed for calculating entire output feature map
	h_base = (blockIdx.z / W_grid) * TILE_WIDTH;//TILE's index in output feature map
	// horizontal base out data index for the block
	w_base = (blockIdx.z % W_grid) * TILE_WIDTH;//TILE's index in output feature map
	// h0 and w0 used as shorthand for threadIdx.x and threadIdx.y
	h0 = threadIdx.x;//index in TILE
	w0 = threadIdx.y;//index in TILE
	h = h_base + h0;//real index in output feature map
	w = w_base + w0;//real index in output feature map
	//h and w is not center point, it's upper left corner point of Input image

	float acc = 0;
	int i, j, p, q;

	// load tile from X[n, c, ...] into shared memory
	for (i = h; i < h_base + X_tile_width; i += TILE_WIDTH)//cuz, data loading
		for (j = w; j < w_base + X_tile_width; j += TILE_WIDTH)
			//h, w are in output range, but +TILE_WIDTH so they can cover input range
			//these for -> loading one pixel in the X_tiles by moving TILE
		{
			if ((h < H_in) && (w < W_in))
				X_shared[(i - h_base)*X_tile_width + j - w_base] = X[n*(M*H_in*W_in)+ m*(H_in*W_in) + (i)*(W_in) + (j)];
		}
	__syncthreads();
	for (p = 0; p < pool_size; p++)
		for (q = 0; q < pool_size; q++)
			if ((h < H_out) && (w < W_out) && ((pool_size*h + p) < X_tile_width) && ((pool_size*w + q) < X_tile_width))
				acc = acc + X_shared[(pool_size*h + p)*X_tile_width + pool_size*w + q]/(pool_size*pool_size);
	//h0, w0 are in TILE's range, but for-loop until K, they can cover X_shared's range
	__syncthreads();

	if ((h < H_out) && (w < W_out))
		Y[n*(M*H_out*W_out) + m*(H_out*W_out) + h*(W_out) + w] = acc;
	//}
#ifdef DEBUG_POOLING_
		if(n==7 && m == 5 && w==13) printf("h=%d, w=%d, acc=%.3lf\n", h, w, acc);
#endif

}

//double for loop version
//M: number of input, output feature maps
//H_in: height of each input image
//W_in: width of each input map image
//X: input feature maps
//Y: output feature maps
__global__ void poolingLayer_backward_GPU(float* X, int H_in, int W_in, float* Y, int M, int pool_size)
{
	int n, m, h, w, p, q;
	int H_out = H_in/pool_size;
	int W_out = W_in/pool_size;
	int W_grid = ceilf((float)W_out/TILE_WIDTH);
	if(W_grid==0)
		W_grid = 1;
	n = blockIdx.x;
	m = blockIdx.y;
	h = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y;
	w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;

	//h and w is not center point of calculating, it's upper left corner point of Input image
	float acc = 0;
	for (p = 0; p < pool_size; p++) { // loop over KxK input samples
		for (q = 0; q < pool_size; q++)
			if(h < H_out && w < W_out)
				X[n*(M*H_in*W_in)+ m*(H_in*W_in) + (pool_size * h + p)*(W_in) + (pool_size * w + q)] =
						Y[n*(M*H_out*W_out)+ m*(H_out*W_out) + h*(W_out) + w];
	}
	__syncthreads();

}

//no loop version
/*
//M:input feature maps == Output feature maps
//H_out: height of each Output map image(backprop. POOL::Output)
//W_out: width of each Output map image(backprop. POOL::Output)
//X: output feature maps(backprop. POOL::Output)
//Y: input feature maps(backprop. POOL::X)
__global__ void poolingLayer_backward_GPU_naive(float* X, int H_out, int W_out, float* Y, int M)
{
	int n, m, h, w, p, q, h_in_output, w_in_output;
	int H_in = H_out*2;
	int W_in = W_out*2;
	int W_grid = ceilf((float)W_in/TILE_WIDTH);
	if(W_grid==0)
		W_grid = 1;
	n = blockIdx.x;
	m = blockIdx.y;
	h = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y;
	w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;
	h_in_output = ceilf((float)h/2) - (h%2); // index in X, for h
	w_in_output = ceilf((float)w/2) - (w%2); // index in X, for w
	//h and w is not center point of calculating, it's upper left corner point of Input image

	if(h < H_in && w < W_in)
		Y[n*(M*H_in*W_in)+ m*(H_in*W_in) + h*(W_in) + w] =
				X[n*(M*H_out*W_out)+ m*(H_out*W_out) + h_in_output*(W_out) + w_in_output];

	//__syncthreads();
}*/
