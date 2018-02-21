#include "Net_h.h"

void FullyConnect::init(int X_h,// X -> minibatch*sequance_data
			int X_w_W_h,
			int W_w)
{
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0,0.1);

	this->X_width = X_w_W_h;
	this->X_height = X_h;
	this->XT_width = X_h;
	this->XT_height=X_w_W_h;
	this->W_width=W_w;
	this->W_height=X_w_W_h;
	this->WT_width=X_w_W_h;
	this->WT_height=W_w;
	this->Output_width=W_w;
	this->Output_height=X_h;
	this->OutputT_width=X_h;
	this->OutputT_height=W_w;

	this->X.resize(X_width*X_height, 0); // when back, dE_dX
	this->X_c.resize(X_width*X_height, 0); // when back, dE_dX
	this->XT.resize(X_height*X_width, 0);
	this->W.resize(W_height*W_width, 0);
	this->W_c.resize(W_width*W_height, 0);
	for(int i=0; i<W_width*W_height ; i++){this->W[i] = distribution(generator);}
	this->WT.resize(W_width*W_height, 0);
	this->b.resize(Output_width, 0.1);
	this->b_c.resize(Output_width, 0.1);
	//for(int i=0; i<Output_width ; i++){this->b[i] = distribution(generator);}
	this->Wgrad.resize(W_width*W_height, 0);// dE_dW
	this->Wgrad_c.resize(W_width*W_height, 0);
	this->Output.resize(X_height*W_width, 0); // when back, dE_dY
	this->Output_c.resize(X_height*W_width, 0); // when back, dE_dY
}

void FullyConnect::forward()
{
	dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH);
	dim3 numBlocks(ceil((float)Output_width/TILE_WIDTH),ceil((float)Output_height/TILE_WIDTH));//bx = O_WIDTH, by = O_HEIGH

	float* X_pointer = thrust::raw_pointer_cast( X.data() );
	float* W_pointer = thrust::raw_pointer_cast( W.data() );
	float* Output_pointer = thrust::raw_pointer_cast( Output.data() );
	float* b_pointer = thrust::raw_pointer_cast( b.data() );
	gemm_with_bias_h<<<numBlocks,threadsPerBlock>>>(X_pointer, W_pointer, Output_pointer, b_pointer, X_height, X_width, W_width, Output_height, Output_width);
}

void FullyConnect::backward()
{
	dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH);
	float* XT_pointer = thrust::raw_pointer_cast( XT.data() );
	float* X_pointer = thrust::raw_pointer_cast( X.data() );
	float* Output_pointer = thrust::raw_pointer_cast( Output.data() );
	float* Wgrad_pointer = thrust::raw_pointer_cast( Wgrad.data() );
	float* W_pointer = thrust::raw_pointer_cast( W.data() );
	float* b_pointer = thrust::raw_pointer_cast( b.data() );

//// fc Wgrad
	transposeMatrix(XT, X, X_height, X_width);
	dim3 numBlocks_back_dE_dW(ceil((float)W_width/TILE_WIDTH), ceil((float)W_height/TILE_WIDTH));//bx = O_WIDTH, by = O_HEIGH

	//cout << "FC backward numBlocks_transpose_X end" << endl << fflush;
	gemm_h<<<numBlocks_back_dE_dW,threadsPerBlock>>>(XT_pointer, Output_pointer, Wgrad_pointer, XT_height, XT_width, Output_width, W_height, W_width);
	//cout << "FC backward numBlocks_back_dE_dW end" << endl << fflush;

//// fc Xgrad
	transposeMatrix(WT, W, W_height, W_width);
	//cout << "FC backward numBlocks_transpose_W end" << endl << fflush;
	float* WT_pointer = thrust::raw_pointer_cast( WT.data() );
	dim3 numBlocks_back_dE_dX(ceil((float)X_width/TILE_WIDTH), ceil((float)X_height/TILE_WIDTH));//bx = O_WIDTH, by = O_HEIGH
	gemm_h<<<numBlocks_back_dE_dX,threadsPerBlock>>>(Output_pointer, WT_pointer, X_pointer, Output_height, Output_width, WT_width, X_height, X_width);
	//cout << "FC backward numBlocks_back_dE_dX end" << endl << fflush;

//// fc Bgrad
	reduceTofirstindex(Output,Output_height,Output_width);

//// gradient descent
	//bx*tx = idata_width*idata*height
	thrust::transform(Wgrad.begin(), Wgrad.end(), Wgrad.begin(), div_h());
	int blockDim_W = ceil((float)W_width*W_height/1024);
	grad_descent<<<blockDim_W,1024>>>(W_pointer, Wgrad_pointer, W_width*W_height);

	thrust::transform(Output.begin(), Output.end(), Output.begin(), div_h());
	int blockDim_b = ceil((float)Output_width/1024);
	grad_descent<<<blockDim_b,1024>>>(b_pointer, Output_pointer, Output_width);

}

