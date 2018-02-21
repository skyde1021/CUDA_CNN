################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../Convolution.cu \
../FullyConnect.cu \
../Functions.cu \
../Mnist.cu \
../Net_GPU_Gemm.cu \
../Net_GPU_Naive.cu \
../Net_GPU_Tiled.cu \
../Net_GPU_test.cu \
../Net_cpu.cu \
../Pool.cu \
../Softmax.cu 

CU_DEPS += \
./Convolution.d \
./FullyConnect.d \
./Functions.d \
./Mnist.d \
./Net_GPU_Gemm.d \
./Net_GPU_Naive.d \
./Net_GPU_Tiled.d \
./Net_GPU_test.d \
./Net_cpu.d \
./Pool.d \
./Softmax.d 

OBJS += \
./Convolution.o \
./FullyConnect.o \
./Functions.o \
./Mnist.o \
./Net_GPU_Gemm.o \
./Net_GPU_Naive.o \
./Net_GPU_Tiled.o \
./Net_GPU_test.o \
./Net_cpu.o \
./Pool.o \
./Softmax.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O3 -optf /home/hee/cuda-workspace/CNN_MM_backup/flag_option -gencode arch=compute_60,code=sm_60  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 -optf /home/hee/cuda-workspace/CNN_MM_backup/flag_option --compile --relocatable-device-code=false -gencode arch=compute_60,code=compute_60 -gencode arch=compute_60,code=sm_60  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


