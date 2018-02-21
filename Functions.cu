#include "Net_h.h"

/***** Function definitions ***************************/


//result to output
void forward_relu(device_vector<float>& input, device_vector<float>& output)
{

	int size_in = input.size();
	float* input_pointer = thrust::raw_pointer_cast(input.data());
	float* output_pointer = thrust::raw_pointer_cast(output.data());
	int block_size = ceil((double)size_in/1024);
	relu_h<<< block_size,1024>>>(input_pointer, output_pointer, size_in);

}

//result to input
void backward_relu(device_vector<float>& input, device_vector<float>& output)
{
	int size_in = input.size();
	float* input_pointer = thrust::raw_pointer_cast(input.data());
	float* output_pointer = thrust::raw_pointer_cast(output.data());
	int block_size = ceil((double)size_in/1024);
	backward_relu_h<<< block_size,1024>>>(input_pointer, output_pointer, size_in);
}

//blocknumber -> size_in/1024
__global__ void relu_h(float* X, float* Y, int size_in)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;

	if(t < size_in)
	{
		if(X[t] < 0) Y[t] = (float)0;
		else Y[t] = X[t];
	}
}

//blocknumber -> size_in/1024
__global__ void backward_relu_h(float* X, float* Y, int size_in)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;

	if(t < size_in)
	{
		if(X[t] < 0) X[t] = (float)0;
		else X[t] = Y[t];
	}
}

//reducing W_in(H_in -> 1)
void reduceTofirstindex(device_vector<float>& input, int H_in, int W_in)
{
	float* input_pointer = thrust::raw_pointer_cast(input.data());
	int block_size = ceil((float)W_in/1024);
	reduceTofirstindex_h<<< block_size,1024>>>(input_pointer, H_in, W_in);
	gpuErrchk( cudaDeviceSynchronize() );
}

void reduceTofirstindex(float* input_pointer, int H_in, int W_in)
{
	int block_size = ceil((float)W_in/1024);
	reduceTofirstindex_h<<< block_size,1024>>>(input_pointer, H_in, W_in);
	gpuErrchk( cudaDeviceSynchronize() );
}

//blocknumber -> Input_width/1024
__global__ void reduceTofirstindex_h(float* X, int H_in, int W_in)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x; // t == width_in
	float tmp=0;
	if(t < W_in)
	{
		for(int i=0 ; i<H_in ; i++)
		{
			tmp += X[i*W_in + t];
		}
		X[0*W_in + t] = tmp;
	}
}

void forward_bias_per_channel(device_vector<float>& input, device_vector<float>& bias, int N, int ch_in, int h_in, int w_in)
{

	dim3 blockDim(N, ch_in, h_in);
	float* input_pointer = thrust::raw_pointer_cast(input.data());
	float* bias_pointer = thrust::raw_pointer_cast(bias.data());
	forward_bias<<< blockDim ,w_in >>>(input_pointer, bias_pointer, N, ch_in, h_in, w_in);
}

//blockDim -> minibatch(N), inputchannel(ch_in), inputheight(h_in)
//threadNum -> input_width(w_in)
__global__ void forward_bias(float* X, float* b, int N, int ch_in, int h_in, int w_in)
{
	int n = blockIdx.x;
	int ch = blockIdx.y;
	int h = blockIdx.z;
	int w =threadIdx.x;
	//__shared__ float shmem[(TILE_WIDTH); // 2 is pooling size

	X[n*(ch_in*h_in*w_in)+ch*(h_in*w_in)+h*(w_in)+w] =
			X[n*(ch_in*h_in*w_in)+ch*(h_in*w_in)+h*(w_in)+w] + b[ch];
}

//w_total_in: w_ch*w_width_mul_w_height
void backward_bias_per_channel(device_vector<float>& input, device_vector<float>& bias, int N, int h_in, int w_total_in, int w_ch, int w_width_mul_w_height)
{
	device_vector<float> input_tmp, input_per_minibatch_T;
	input_tmp.resize(h_in*w_total_in,0);
	input_per_minibatch_T.resize(h_in*w_total_in,0);
	input_tmp = input;

	float* input_pointer = thrust::raw_pointer_cast(input_tmp.data());
	float* input_per_minibatch_T_pointer = thrust::raw_pointer_cast(input_per_minibatch_T.data());
	float* bias_pointer = thrust::raw_pointer_cast(bias.data());

	for(int i=0; i<N ;i++)
	{
		//N*(C*H*W) -> N*(H*W*C)
		transposeMatrix(input_per_minibatch_T_pointer, input_pointer, w_ch, w_width_mul_w_height);
		//reduceTofirstchannel
		reduceTofirstindex(input_per_minibatch_T_pointer, w_width_mul_w_height, w_ch);
		input_pointer += w_total_in;
		input_per_minibatch_T_pointer += w_total_in;
	}

	input_per_minibatch_T_pointer = thrust::raw_pointer_cast(input_per_minibatch_T.data());
	//reduceTofirstminibatch
	reduceTofirstindex(input_per_minibatch_T_pointer, h_in, w_total_in);

	//grad descent
	int blockDim_b = ceil((float)w_ch/1024);
	thrust::transform(input_per_minibatch_T.begin(),
			input_per_minibatch_T.end(), input_per_minibatch_T.begin(), div_h());
	grad_descent<<<blockDim_b, 1024>>>(bias_pointer, input_per_minibatch_T_pointer, w_ch);
}



//size_in -> entire width of vector(MINIBATCH*Outputimage_channel*Outputimage_height*Outputimage_width)
void sigmoid(device_vector<float>& input, int size_in)
{
	float* input_pointer = thrust::raw_pointer_cast(input.data());
	int block_size= ceil((float)size_in/1024);
	sigmoid<<< block_size,1024>>>(input_pointer, size_in);
}


//blocknumber -> Input_width/1024
__global__ void sigmoid(float* X, int size_in)
{
	int t = blockIdx.x * 1024 + threadIdx.x;

	if(t < size_in)
		X[t] = 1 / (1 + exp(-X[t]));
}

//blocknumber -> Input_width/1024
__global__ void backward_sigmoid(float* X, int size_in)
{
	int t = blockIdx.x * 1024 + threadIdx.x;

	if(t < size_in)
	{
		double tmp = 1 / (1 + exp(-X[t]));
		tmp = (1-tmp)*tmp;
		X[t] = X[t]*tmp;
	}
}

//size_in -> entire width of vector(MINIBATCH*Outputimage_channel*Outputimage_height*Outputimage_width)
void backward_sigmoid(device_vector<float>& input, int size_in)
{
	float* input_pointer = thrust::raw_pointer_cast(input.data());
	int block_size= ceil((float)size_in/1024);
	backward_sigmoid<<< block_size,1024>>>(input_pointer, size_in);
}

//blocknumber -> Input_width/1024
__global__ void div_by_constant(float* X, int n, int size_in)
{
	int t = blockIdx.x * 1024 + threadIdx.x;

	X[t] = X[t]/n;
}

//size_in -> entire width of vector(MINIBATCH*Outputimage_channel*Outputimage_height*Outputimage_width)
void div_by_constant(device_vector<float>& input, int n, int size_in)
{
	float* input_pointer = thrust::raw_pointer_cast(input.data());
	int block_size= ceil((float)size_in/1024);
	div_by_constant<<< block_size,1024>>>(input_pointer , n, size_in);
}

int reverse_int32 (int i){
	unsigned char byte1, byte2, byte3, byte4;
	byte1 = i&MAXBYTE;
	byte2 = (i>>8)&MAXBYTE;
	byte3 = (i>>16)&MAXBYTE;
	byte4 = (i>>24)&MAXBYTE;
	return ( (int)byte1<<24 ) + ( (int)byte2<<16 ) + ( (int)byte3<<8 ) + (int)byte4;
}
/*
	Read [number_of_images]x28x28 MNIST data from {datapath}
	Store data into the given float array
*/
void read_data(const char* datapath, host_vector< host_vector<float> >& data){

	ifstream infile (datapath, ios::binary);
	if (!infile.is_open())
	{
		printf("FAILED TO OPEN FILE: %s\n", datapath);
		return ;
	}
	cout << "== Input test image file: " << datapath << endl;
	// read the header information
	int magic_number=0;
	int number_of_images=0;
	int n_rows = 0;
	int n_cols = 0;
	infile.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverse_int32(magic_number);
	cout << "magic number: " << magic_number << endl;

	infile.read((char*)&number_of_images, sizeof(number_of_images));
	number_of_images = reverse_int32(number_of_images);
	cout << "number of images: " << number_of_images << endl;

	infile.read((char*)&n_rows, sizeof(n_rows));
	n_rows = reverse_int32(n_rows);

	infile.read((char*)&n_cols, sizeof(n_cols));
	n_cols = reverse_int32(n_cols);

	n_rows += 4;
	n_cols += 4;

	cout << "size of row = " << n_rows << ", size of cols = " << n_cols << endl;

	// Read actual data (uint8 -> float)
	for(int i=0; i<number_of_images/MINIBATCH; ++i)
	{
		for(int n=0 ; n<MINIBATCH ; n++)
		{
			for(int r=2; r<n_rows-2; ++r)
			{
				for(int c=2; c<n_cols-2; ++c)
				{
					unsigned char temp = 0;
					infile.read((char*)&temp, sizeof(temp));
#ifdef DEBUG_H_
					data[i][(n_cols*n_rows*n)+(n_rows*r)+c] = r+c;

#else
					data[i][(n_cols*n_rows*n)+(n_rows*r)+c] = (float)temp * (float)MNIST_SCALE_FACTOR;
#endif
				}
			}
		}
	}
	infile.close();
	cout << "Done. [data: "<<datapath <<"] [count: " << number_of_images <<"]"<<endl;
}

/*
	Read [number_of_images]x28x28 MNIST data from {datapath}
	Store data into the given float array
*/
void read_data_no_padding(const char* datapath, host_vector< host_vector<float> >& data){

	ifstream infile (datapath, ios::binary);
	if (!infile.is_open())
	{
		printf("FAILED TO OPEN FILE: %s\n", datapath);
		return ;
	}
	cout << "== Input test image file: " << datapath << endl;
	// read the header information
	int magic_number=0;
	int number_of_images=0;
	int n_rows = 0;
	int n_cols = 0;
	infile.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverse_int32(magic_number);
	cout << "magic number: " << magic_number << endl;

	infile.read((char*)&number_of_images, sizeof(number_of_images));
	number_of_images = reverse_int32(number_of_images);
	cout << "number of images: " << number_of_images << endl;

	infile.read((char*)&n_rows, sizeof(n_rows));
	n_rows = reverse_int32(n_rows);

	infile.read((char*)&n_cols, sizeof(n_cols));
	n_cols = reverse_int32(n_cols);

	cout << "size of row = " << n_rows << ", size of cols = " << n_cols << endl;

	// Read actual data (uint8 -> float)
	for(int i=0; i<number_of_images/MINIBATCH; ++i)
	{
		//printf("%d index\n", i);fflush(stdout);
		for(int n=0 ; n<MINIBATCH ; n++)
		{
			//data[i][0] = 1;
			//for(int r=0; r<n_rows; ++r)
			for(int r=0; r<n_rows; ++r)
			{
				//for(int c=0; c<n_cols; ++c)
				for(int c=0; c<n_cols; ++c)
				{
					unsigned char temp = 0;
					infile.read((char*)&temp, sizeof(temp));
#ifdef DEBUG_H_
					data[i][(n_cols*n_rows*n)+(n_rows*r)+c] = r+c;

#else
					data[i][(n_cols*n_rows*n)+(n_rows*r)+c] = (float)temp * (float)MNIST_SCALE_FACTOR;
#endif
				}
			}
		}
	}
	infile.close();
	cout << "Done. [data: "<<datapath <<"] [count: " << number_of_images <<"]"<<endl;
}


void read_label(const char* labelPath, host_vector<int>& labels){
	int number_of_labels = 0;

	ifstream infile(labelPath, ios::binary);
	if (!infile.is_open())
	{
			printf("FAILED TO OPEN FILE: %s\n", labelPath);
			return;
	}
	cout << "== Input test label file: " << labelPath << endl;

	int magic_number=0;
	// read the label information
	infile.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverse_int32(magic_number);
	cout << "magic number: " << magic_number << endl;

	infile.read((char*)&number_of_labels, sizeof(number_of_labels));
		number_of_labels = reverse_int32(number_of_labels);
	cout << "number of labels: " << number_of_labels << endl;


	for(int i=0; i<number_of_labels; ++i)
	{
		unsigned char temp = 0;
		infile.read((char*)&temp, sizeof(temp));
		labels[i] = (int)temp;

	}
	infile.close();
	cout << "Done. [data: "<<labelPath <<"] [count: " << number_of_labels<<"] "<<endl;
}

__global__ void gemm_h(float* Md, float* Nd, float* Pd, int M_height_in, int M_width_N_height_in, int N_width_in , int height_out, int width_out)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;

	float Pvalue = 0;
	//width
	for(int m = 0 ; m < ceilf((float)M_width_N_height_in / TILE_WIDTH) ; ++m)
	{
		if(row < M_height_in && (m*TILE_WIDTH + tx) < M_width_N_height_in) // X
			Mds[ty][tx] = Md[row*M_width_N_height_in+(m*TILE_WIDTH + tx)];
		else
			Mds[ty][tx] = 0;
		if((m*TILE_WIDTH + ty) < M_width_N_height_in && col < N_width_in) // W
			Nds[ty][tx] = Nd[(m*TILE_WIDTH + ty)*N_width_in + col];
		else
			Nds[ty][tx] = 0;
		__syncthreads();

		for(int k = 0 ; k < TILE_WIDTH ; ++k)
		{
			Pvalue += Mds[ty][k] * Nds[k][tx];
		}

		__syncthreads();
	}

	if(row < height_out && col < width_out)
		Pd[row*width_out + col] = Pvalue; // Output
}

__global__ void gemm_with_bias_h(float* Md, float* Nd, float* Pd, float* B, int M_height_in, int M_width_N_height_in, int N_width_in , int height_out, int width_out)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;

	float Pvalue = 0;

	//width
	for(int m = 0 ; m < ceilf((float)M_width_N_height_in / TILE_WIDTH) ; ++m)
	{
		if(row < M_height_in && (m*TILE_WIDTH + tx) < M_width_N_height_in) // X
			Mds[ty][tx] = Md[row*M_width_N_height_in+(m*TILE_WIDTH + tx)];
		else
			Mds[ty][tx] = 0;
		if((m*TILE_WIDTH + ty) < M_width_N_height_in && col < N_width_in) // W
			Nds[ty][tx] = Nd[(m*TILE_WIDTH + ty)*N_width_in + col];
		else
			Nds[ty][tx] = 0;
		__syncthreads();

		for(int k = 0 ; k < TILE_WIDTH ; ++k)
		{
			Pvalue += Mds[ty][k] * Nds[k][tx];
		}

		__syncthreads();
	}

	if(row < height_out && col < width_out)
		Pd[row*width_out + col] = Pvalue + B[col]; // Output
}

void transposeMatrix(device_vector<float>& outputT, device_vector<float>& input, int input_height, int input_width)
{
	dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH);
	dim3 numBlocks_transpose_X(ceil((float)input_width/TILE_WIDTH), ceil((float)input_height/TILE_WIDTH));//bx = input_WIDTH, by = input_HEIGHT
	float* XT_pointer = thrust::raw_pointer_cast( outputT.data() );
	float* X_pointer = thrust::raw_pointer_cast( input.data() );
	transposeMatrix_h<<<numBlocks_transpose_X,threadsPerBlock>>>(XT_pointer, X_pointer, input_height, input_width);
}

void transposeMatrix(float* XT_pointer, float* X_pointer, int input_height, int input_width)
{
	dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH);
	dim3 numBlocks_transpose_X(ceil((float)input_width/TILE_WIDTH), ceil((float)input_height/TILE_WIDTH));//bx = input_WIDTH, by = input_HEIGHT
	transposeMatrix_h<<<numBlocks_transpose_X,threadsPerBlock>>>(XT_pointer, X_pointer, input_height, input_width);
}

//input shape dependency(gridDim -> input Matrix dimension)
__global__ void transposeMatrix_h(float *odata, const float *idata, int height_in, int width_in)
{
  int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int y = blockIdx.y * TILE_WIDTH + threadIdx.y;

  if(y < height_in && x < width_in)
	  odata[(x)*height_in + y] = idata[(y)*width_in + x];
  __syncthreads();
}

__global__ void grad_descent(float *odata, const float *idata, int size)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if(t < size)
	{
		float tmp = odata[t];
		tmp = tmp - (float)LEARNIG_RATE*idata[t];
		odata[t] = tmp;
	}
}

__global__ void init_zero_vector(float *X, int size_in)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;

	if(t < size_in)
	{
		X[t] = (float)0;
	}
}

void printMNIST(host_vector<float>& data, int after_minib){
	int start_index = after_minib*RAW_DIM_PADDING*RAW_DIM_PADDING;
	//cout << "start" << start_index << " size " << data.size() << flush;
	for(int r=0; r<RAW_DIM_PADDING; r++){
		for (int c=0; c<RAW_DIM_PADDING; c++){
			if (data[start_index+r*RAW_DIM_PADDING+c] > 0.5f){
				cout << "■" << flush;
			}
			else {
				cout << "□" << flush;
			}
		}
		cout << endl << flush;
	}
}

void printMNIST_num(host_vector<float>& data, int label, int num){
	cout << "Check data for label " << label << endl;
	for(int r=0; r<RAW_DIM_PADDING*num; r++){
		for (int c=0; c<RAW_DIM_PADDING; c++){
			if (data[r*RAW_DIM_PADDING+c] > 0.5f){
				cout << "■" << flush;
			}
			else {
				cout << "□" << flush;
			}
		}
		cout << endl << flush;
	}
}

void printMNIST_HW_row_col(device_vector<float>& data_tmp,int height, int width, int row_start_index, int col_start_index,
		int row_num, int col_num, int row_interval, int col_interval, char* str){
#ifdef DEBUG_PRINT_ROW_COL1
	host_vector<float> data;
	data.resize(data_tmp.size());
	data = data_tmp;
	cout << endl << flush;cout << endl << flush;
	printf(str);
	printf("\ndata size: %d, height: %d, width: %d\n", data.size(), height, width);
	float* data_pointer = thrust::raw_pointer_cast(data.data());
	if(row_num == 0 || col_num == 0)
	{
		for(int r=0; r<height; r++){
			for (int c=0; c<width; c++){
				printf("%.8lf ", data_pointer[r*width+c]);
			}
			cout << endl << flush;
		}
	}else
	{
		for(int r=row_start_index; r<row_start_index+(row_num*row_interval); r+=row_interval){
			for (int c=col_start_index; c<col_start_index+(col_num*col_interval); c+=col_interval){
				printf("%.8lf ", data_pointer[r*width+c]);
			}
			cout << endl << flush;
		}
	}
#endif
}

void printMNIST_HW_avg_value(host_vector<float>& data, char* str)
{
	printf(str);
	printf(" data size: %d\n", data.size());
	float* data_pointer = thrust::raw_pointer_cast(data.data());
	double sum = 0;
	double var = 0;
	double avg = 0;
	double avg_positive = 0;
	double avg_negative = 0;
	int data_size = data.size();

	for(int r=0; r<data_size; r++){
			float tmp = data_pointer[r];
			if(tmp < 0)sum += -tmp;else sum += tmp;
			if(tmp < 0)avg_negative += tmp;else avg_positive += tmp;
	}
	avg = sum/data_size;
	avg_negative = avg_negative/data_size;
	avg_positive = avg_positive/data_size;
	for(int r=0; r<data_size; r++){
			float tmp = data_pointer[r];
			if(tmp < 0)var += pow(-tmp-avg,2);else var += pow(tmp-avg,2);
	}
	printf("abavg<%.8lf> ", avg);
	printf("stddev<%.8lf> ", sqrt(var/data_size));
	printf("+avg<%.8lf> ", avg_positive);
	printf("-avg<%.8lf> ", avg_negative);
	cout << endl << flush;
}

void printMNIST_HW_avg_value(device_vector<float>& data_tmp, char* str)
{
	host_vector<float> data;
	data.resize(data_tmp.size());
	data = data_tmp;
	printf(str);
	printf(" data size: %d\n", data.size());
	float* data_pointer = thrust::raw_pointer_cast(data.data());
	double sum = 0;
	double var = 0;
	double avg = 0;
	double avg_positive = 0;
	double avg_negative = 0;
	int data_size = data_tmp.size();

	for(int r=0; r<data_size; r++){
			float tmp = data_pointer[r];
			if(tmp < 0)sum += -tmp;else sum += tmp;
			if(tmp < 0)avg_negative += tmp;else avg_positive += tmp;
	}
	avg = sum/data_size;
	avg_negative = avg_negative/data_size;
	avg_positive = avg_positive/data_size;
	for(int r=0; r<data_size; r++){
			float tmp = data_pointer[r];
			if(tmp < 0)var += pow(-tmp-avg,2);else var += pow(tmp-avg,2);
	}
	printf("abavg<%.8lf> ", avg);
	printf("stddev<%.8lf> ", sqrt(var/data_size));
	printf("+avg<%.8lf> ", avg_positive);
	printf("-avg<%.8lf> ", avg_negative);
	cout << endl << flush;
}

void clear()
{
	usleep(100000);
	for(int i=0; i<100;i++)printf("\n");
}
