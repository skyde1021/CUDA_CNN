#include "Net_h.h"

const char* PATH_TRAIN_DATA = "../mnist/train-images-idx3-ubyte";
const char* PATH_TRAIN_LABEL = "../mnist/train-labels-idx1-ubyte";
const char* PATH_TEST_DATA = "../mnist/t10k-images-idx3-ubyte";
const char* PATH_TEST_LABEL = "../mnist/t10k-labels-idx1-ubyte";

/***** Main function ***********************************/

int main(){
	
	host_vector< host_vector<float> > Xtrain(NUM_TRAIN/MINIBATCH, host_vector<float>(RAW_PIXELS_PER_IMG_PADDING*MINIBATCH, 0));
	host_vector< host_vector<float> > Xtest(NUM_TEST/MINIBATCH, host_vector<float>(RAW_PIXELS_PER_IMG_PADDING*MINIBATCH, 0));
	host_vector<int> Ytrain(NUM_TRAIN, 0);
	host_vector<int> Ytest(NUM_TEST, 0);
	int checkLabel;

//// load data
	read_data (PATH_TRAIN_DATA, Xtrain);
	read_label(PATH_TRAIN_LABEL, Ytrain);
	
	read_data (PATH_TEST_DATA, Xtest);
	read_label(PATH_TEST_LABEL, Ytest);
/*
//// check data (optional)
	checkLabel = 0;
	//relu(1, 32, Xtrain);
	printMNIST(Xtrain[checkLabel], Ytrain[checkLabel]);
	printMNIST(Xtest[checkLabel], Ytest[checkLabel]);
	checkLabel = 59999/MINIBATCH;
	printMNIST(Xtrain[checkLabel], Ytrain[checkLabel]);
	checkLabel = 9999/MINIBATCH;
	printMNIST(Xtest[checkLabel], Ytest[checkLabel]);*/

	Net_GPU_Tiled l_gpu_tiled;
	l_gpu_tiled.train(Xtrain, Ytrain);
	l_gpu_tiled.test(Xtest, Ytest);

	Net_GPU_Naive l_gpu_naive;
	l_gpu_naive.train(Xtrain, Ytrain);
	l_gpu_naive.test(Xtest, Ytest);

	Net_GPU_Gemm l_gpu_gemm;
	l_gpu_gemm.train(Xtrain, Ytrain);
	l_gpu_gemm.test(Xtest, Ytest);

	Net_CPU lenet_l;
	lenet_l.train(Xtrain, Ytrain);
	lenet_l.test(Xtest, Ytest);

	//Net_GPU_test l_gpu_test;
	//l_gpu_test.train(Xtrain, Ytrain);
	//l_gpu_test.test(Xtest, Ytest);

}
