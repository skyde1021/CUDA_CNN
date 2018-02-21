#include "Net_h.h"


Net_GPU_Tiled::Net_GPU_Tiled()
{
	//// conv1 Input = 1x32x32 Output = 8x20x20
	//// pool1 Input = 8x20x20 Output = 8x5x5

	//// FC1 Input = 8x5x5(200) Output = 200
	//// FC2 Input = 200 Output = 200
	//// FC3 Input = 200 Output = 10


	/*conv1.init(MINIBATCH, 32, 32, 1, 5, 6);
	pool1.init(MINIBATCH, 28, 28, 6,4);

	fc1.init(MINIBATCH, 7*7*6, 200);*/
	conv1.init(MINIBATCH, 32, 32, 1, CONV_KERNEL_SIZE, 8);
	pool1.init(MINIBATCH, 20, 20, 8,4);

	fc1.init(MINIBATCH, 5*5*8, 200);
	fc2.init(MINIBATCH, 200, 200);
	fc3.init(MINIBATCH, 200, 10);
	sm1.delta_c.resize(MINIBATCH*10, 0);
	correct_num = 0;
}

Net_GPU_Tiled::~Net_GPU_Tiled()
{

}

void Net_GPU_Tiled::train(host_vector< host_vector<float> >& Xtrain, host_vector<int>& Ytrain)
{
	dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH);
	cout << "Net_GPU_Tiled Train Start" << endl << fflush;
	int minibatch_index = 0;

	struct timespec start, finish;
	double elapsed=0;
	correct_num=0;
	int fine_epoch=0;
	float pre_acc=0;
	float max_acc=0;


//// MxCxHxW M=MINIBATCH

//// CxHxW

	for(int epoch=1 ; epoch <= 100 ; epoch++){
		minibatch_index=0;
		correct_num=0;
		elapsed=0;

		while(minibatch_index < NUM_TRAIN/MINIBATCH)
		{
			clock_gettime(CLOCK_REALTIME, &start);
			conv1.X = Xtrain[minibatch_index];

			conv1.forward_GPU_tiled();

			pool1.forward_GPU_tiled(conv1.Output);
			forward_bias_per_channel(pool1.Output, pool1.b, MINIBATCH,
					pool1.Outputimage_channel, pool1.Outputimage_height, pool1.Outputimage_width);

			forward_relu(pool1.Output, fc1.X);

			fc1.forward();

			forward_relu(fc1.Output, fc2.X);

			fc2.forward();

			forward_relu(fc2.Output, fc3.X);

			//fc3.X = fc2.Output;
			fc3.forward();
			fc3.Output_c = fc3.Output;

		//// SoftMax Input = 10 Output = 10
			sm1.accuracy(MINIBATCH, 10, Xtrain, Ytrain, fc3.Output_c, minibatch_index, correct_num);

			sm1.softmax(MINIBATCH, 10, Ytrain, fc3.Output_c);

			sm1.cross_entropy_loss(MINIBATCH, Ytrain, fc3.Output_c, 10, sm1.loss, minibatch_index);

		//// SoftMax delta
			sm1.softmax_backward( MINIBATCH, Ytrain, fc3.Output_c, sm1.delta_c, 10, minibatch_index);

		//// fc backward
			fc3.Output = sm1.delta_c;
			fc3.backward();

			backward_relu(fc2.Output, fc3.X);

			fc2.Output = fc3.X;
			fc2.backward();

			backward_relu(fc1.Output, fc2.X);

			fc1.backward();

			backward_relu(pool1.Output, fc1.X);

			backward_bias_per_channel(pool1.Output, pool1.b, MINIBATCH, pool1.Output_height, pool1.Output_width,
					pool1.Outputimage_channel, pool1.Outputimage_height*pool1.Outputimage_width);

			pool1.backward_GPU(fc1.X);

			conv1.Output = pool1.X;
			conv1.backward_GPU_gemm();

			clock_gettime(CLOCK_REALTIME, &finish);

			elapsed += ((double)finish.tv_sec-start.tv_sec) + ((double)finish.tv_nsec - start.tv_nsec)/ 1000000000.0;

			minibatch_index += 1;

			if((minibatch_index*MINIBATCH) == 60000)
			{
				/*
								printMNIST_HW_avg_value(sm1.delta_c,"sm1.delta_c");

								printMNIST_HW_avg_value(fc3.W,"fc3 w");
								printMNIST_HW_avg_value(fc3.b,"fc3 b");
								printMNIST_HW_avg_value(fc2.W,"fc2 w");
								printMNIST_HW_avg_value(fc2.b,"fc2 b");
								printMNIST_HW_avg_value(fc1.W,"fc1 w");
								printMNIST_HW_avg_value(fc1.b,"fc1 b");
								printMNIST_HW_avg_value(pool1.b,"pool1 b");
								printMNIST_HW_avg_value(conv1.W,"conv1 w");
				*/
								float acc = (float)correct_num/(60000);
								if(acc > max_acc)
								{
									max_acc = acc;
									fine_epoch=epoch;
								}
								correct_num=0;

								printf("[Epoch %d] minibatch %d (%.7lf images/sec) max_acc %.3f acc %.3f elapsed time %.3f\n",
										epoch, minibatch_index , minibatch_index*MINIBATCH/elapsed, max_acc*100, acc*100, elapsed);fflush(stdin);
			}
		}
	}
}

void Net_GPU_Tiled::test(host_vector< host_vector<float> >& Xtest, host_vector<int>& Ytest)
{
	dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH);
	cout << "Net_GPU_Tiled Test Start" << endl << fflush;
	int minibatch_index = 0;

	struct timespec start, finish;
	double elapsed=0;
	correct_num=0;
	int fine_epoch=0;
	float pre_acc=0;
	float max_acc=0;

//// MxCxHxW M=MINIBATCH

//// CxHxW

	while(minibatch_index < NUM_TEST/MINIBATCH)
	{
		clock_gettime(CLOCK_REALTIME, &start);
		conv1.X = Xtest[minibatch_index];

		conv1.forward_GPU_gemm();

		pool1.forward_GPU_tiled(conv1.Output);
		forward_bias_per_channel(pool1.Output, pool1.b, MINIBATCH,
				pool1.Outputimage_channel, pool1.Outputimage_height, pool1.Outputimage_width);

		forward_relu(pool1.Output, fc1.X);

		fc1.forward();

		forward_relu(fc1.Output, fc2.X);

		fc2.forward();

		forward_relu(fc2.Output, fc3.X);

		//fc3.X = fc2.Output;
		fc3.forward();
		fc3.Output_c = fc3.Output;

	//// SoftMax Input = 10 Output = 10
		sm1.accuracy(MINIBATCH, 10, Xtest, Ytest, fc3.Output_c, minibatch_index, correct_num);

		sm1.softmax(MINIBATCH, 10, Ytest, fc3.Output_c);

		sm1.cross_entropy_loss(MINIBATCH, Ytest, fc3.Output_c, 10, sm1.loss, minibatch_index);

	//// SoftMax delta
		sm1.softmax_backward( MINIBATCH, Ytest, fc3.Output_c, sm1.delta_c, 10, minibatch_index);

		clock_gettime(CLOCK_REALTIME, &finish);

		elapsed += ((double)finish.tv_sec-start.tv_sec) + ((double)finish.tv_nsec - start.tv_nsec)/ 1000000000.0;

		minibatch_index += 1;

		if((minibatch_index*MINIBATCH) == 10000)
		{

			float acc = (float)correct_num/(10000);
			correct_num=0;

			printf("[Test](%.7lf images/sec) acc %.3f elapsed time %.3f\n",
					minibatch_index*MINIBATCH/elapsed, acc*100, elapsed);fflush(stdin);
		}
	}
}

