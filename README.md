# CNN implementation with C++ and CUDA
> Various versions (CPU, CUDA_NAIVE, CUDA_TILED, GEMM) of convolutional neural network implementations  
by Heechul Lim

## Layer configuration
- Convolution (Forward CPU, CUDA_NAIVE, CUDA_TILED, GEMM)  
Input: 1 * 32 * 32  
Output: 1 * 20 * 20  
Kernel size: 13  
Kernel dimension: 8   

- Pooling (Forward CPU, CUDA_NAIVE, CUDA_TILED)  
Input: 8 * 20 * 20  
Output: 8 * 5 * 5  
Kernel size: 4

- Relu (CUDA)  

- Inner prodect 1 (CUDA)  
Input: 8 * 5 * 5 (flatten 200)  
Output: 200

- Relu (CUDA)  

- Inner prodect 2  (CUDA)  
Input: 200  
Output: 200

- Relu (CUDA)  

- Inner prodect 3  (CUDA)  
Input: 200  
Output: 10

- Softmax

**Computational cost of convolution: 80-90% of the total execution**  
(http://on-demand.gputechconf.com/gtc/2015/webinar/gtc-express-deep-learning-with-cuDNN-webinar.pdf) 

## Dataset
- MNIST: 6k train set, 1k test set
- 1 * 32 * 32 (padding 2)

## Accuracy
- 1.3 epoch: 90%
- 30 epoch: 98%  

**It depends on minibatch number and learning rate**

## Experiment environment
- CPU: Xeon E5-2630 v4 @ 2.2Ghz
- GPU: NVIDIA GTX 1080 TI

## Result with training set (6k)
- Minibatch 100  

Name | Elapsed time (1 epoch) | Processing speed (images/sec)
:----: | :----------------------: | :-----------------------------:
CPU | 39.391 | 1523.2
CUDA NAIVE | 5.693 | 10539.9
CUDA TILED | 5.160 | 11628.1
GEMM | 7.890 | 7604.7

- Minibatch 2  

Name | Elapsed time (1 epoch) | Processing speed (images/sec)
:----: | :----------------------: | :-----------------------------:
CPU | 53.303 | 1125.6
CUDA NAIVE | 17.048 | 3519.5
CUDA TILED | 15.877 | 3778.9
GEMM | 18.475 | 3247.6

## Usage
```
cd ./Release
make clean
make
./CNN
``` 

## Reference
- http://eric-yuan.me/cnn/
- Programming Massively Parallel Processors written by David B. Kirk and Wen-mei W. Hwu
