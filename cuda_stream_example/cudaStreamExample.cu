#include <vector>
#include <random>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef DEBUG
#define CUDA_CALL(F)  if( (F) != cudaSuccess) \
    {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__); exit(-1);}
#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
    {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__ - 1); exit(-1);}
#else
#define CUDA_CALL(F) (F)
#define CUDA_CHECK()
#endif

void printDeviceInto();
void generateBGRA8K(uint8_t *buffer, int dataSize);
void convertPixelFormatCPU(uint8_t *inputBGRA, uint8_t *outputYUV, int numPixels);
__global__ void convertPixelFormatCUDA(uint8_t *inputBGRA, uint8_t *outputYUV, int numPixels);

int main()
{
    printDeviceInto();

    uint8_t *BGRABuffer;
    uint8_t *YUVBuffer;
    uint8_t *d_BGRABuffer;
    uint8_t *d_YUVBuffer;
    const int dataSizeBGRA = 7680 * 4320 * 4;
    const int dataSizeYUV = 7680 * 4320 * 3;

    CUDA_CALL(cudaMallocHost(&BGRABuffer, dataSizeBGRA));
    CUDA_CALL(cudaMallocHost(&YUVBuffer, dataSizeYUV));
    CUDA_CALL(cudaMalloc(&d_BGRABuffer, dataSizeBGRA));
    CUDA_CALL(cudaMalloc(&d_YUVBuffer, dataSizeYUV));

    std::vector<uint8_t> YUVCPUBuffer(dataSizeYUV);

    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsedTime;
    float elapsedTimeTotal;
    float dataRate;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    std::cout << " " << std::endl;
    std::cout << "Generating 7680 x 4320 BRGA8888 image, data size: " << dataSizeBGRA << std::endl;
    generateBGRA8K(BGRABuffer, dataSizeBGRA);

    std::cout << " " << std::endl;
    std::cout << "Computing results using CPU." << std::endl;
    std::cout << " " << std::endl;
    CUDA_CALL(cudaEventRecord(start, 0));
    convertPixelFormatCPU(BGRABuffer, YUVCPUBuffer.data(), 7680 * 4320);
    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "    Whole process took " << elapsedTime << "ms." << std::endl;

    std::cout << " " << std::endl;
    std::cout << "Computing results using GPU, default stream." << std::endl;
    std::cout << " " << std::endl;

    std::cout << "    Move data to GPU." << std::endl;
    CUDA_CALL(cudaEventRecord(start, 0));
    CUDA_CALL(cudaMemcpy(d_BGRABuffer, BGRABuffer, dataSizeBGRA, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
    dataRate = dataSizeBGRA / (elapsedTime / 1000.0) / 1.0e9;
    elapsedTimeTotal = elapsedTime;
    std::cout << "        Data transfer took " << elapsedTime << "ms." << std::endl;
    std::cout << "        Performance is " << dataRate << "GB/s." << std::endl;

    std::cout << "    Convert 8-bit BGRA to 8-bit YUV." << std::endl;
    CUDA_CALL(cudaEventRecord(start, 0));
    convertPixelFormatCUDA<<<32400, 1024>>>(d_BGRABuffer, d_YUVBuffer, 7680 * 4320);
    CUDA_CHECK();
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
    dataRate = dataSizeBGRA / (elapsedTime / 1000.0) / 1.0e9;
    elapsedTimeTotal += elapsedTime;
    std::cout << "        Processing of 8K image took " << elapsedTime << "ms." << std::endl;
    std::cout << "        Performance is " << dataRate << "GB/s." << std::endl;

    std::cout << "    Move data to CPU." << std::endl;
    CUDA_CALL(cudaEventRecord(start, 0));
    CUDA_CALL(cudaMemcpy(YUVBuffer, d_YUVBuffer, dataSizeYUV, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize());
    CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop);
    dataRate = dataSizeYUV / (elapsedTime / 1000.0) / 1.0e9;
    elapsedTimeTotal += elapsedTime;
    std::cout << "        Data transfer took " << elapsedTime << "ms." << std::endl;
    std::cout << "        Performance is " << dataRate << "GB/s." << std::endl;

    std::cout << "    Whole process took " << elapsedTimeTotal << "ms." <<std::endl;

    std::cout << "    Compare CPU and GPU results ..." << std::endl;
    bool foundMistake = false;
    for(int i = 0; i < dataSizeYUV; i++)
    {
        if(YUVCPUBuffer[i] != YUVBuffer[i])
        {
            foundMistake = true;
            break;
        }
    }

    if(foundMistake)
    {
        std::cout << "        Results are NOT the same." << std::endl;
    }
    else
    {
        std::cout << "        Results are the same." << std::endl;
    }

    const int nStreams = 16;
    std::cout << " " << std::endl;
    std::cout << "Computing results using GPU, using "<< nStreams <<" streams." << std::endl;
    std::cout << " " << std::endl;
    cudaStream_t streams[nStreams];
    std::cout << "    Creating " << nStreams << " CUDA streams." << std::endl;
    for(int i = 0; i < nStreams; i++)
    {
        CUDA_CALL(cudaStreamCreate(&streams[i]));
    }

    // Notice that data size may not be divided equally
    int BGRAOffset = 0;
    int YUVOffset = 0;
    const int BGRAChunkSize = dataSizeBGRA / nStreams;
    const int YUVChunkSize = dataSizeYUV / nStreams;

    CUDA_CALL(cudaEventRecord(start, 0));
    for(int i = 0; i < nStreams; i++)
    {
        std::cout << "        Launching stream " << i << "." << std::endl;
        BGRAOffset = BGRAChunkSize * i;
        YUVOffset = YUVChunkSize * i;
        CUDA_CALL(cudaMemcpyAsync(d_BGRABuffer + BGRAOffset,
                                  BGRABuffer + BGRAOffset,
                                  BGRAChunkSize,
                                  cudaMemcpyHostToDevice,
                                  streams[i]));

        convertPixelFormatCUDA<<<4096, 1024, 0, streams[i]>>>(d_BGRABuffer + BGRAOffset, d_YUVBuffer + YUVOffset, BGRAChunkSize / 4);

        CUDA_CALL(cudaMemcpyAsync(YUVBuffer + YUVOffset,
                                  d_YUVBuffer + YUVOffset,
                                  YUVChunkSize,
                                  cudaMemcpyDeviceToHost,
                                  streams[i]));
    }

    CUDA_CHECK();
    CUDA_CALL(cudaDeviceSynchronize());
    
    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "    Whole process took " << elapsedTime << "ms." << std::endl;

    std::cout << "    Compare CPU and GPU results ..." << std::endl;
    for(int i=0; i<dataSizeYUV; i++)
    {
        if(YUVCPUBuffer[i] != YUVBuffer[i])
        {
            foundMistake = true;
            break;
        }
    }

    if(foundMistake)
    {
        std::cout << "        Results are NOT the same." << std::endl;
    }
    else
    {
        std::cout << "        Results are the same." << std::endl;
    }

    CUDA_CALL(cudaFreeHost(BGRABuffer));
    CUDA_CALL(cudaFreeHost(YUVBuffer));
    CUDA_CALL(cudaFree(d_BGRABuffer));
    CUDA_CALL(cudaFree(d_YUVBuffer));

    return 0;
}

void printDeviceInto()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Number of device(s): " << deviceCount << std::endl;
    if(deviceCount == 0)
    {
        std::cout << "There is no device supporting CUDA" << std::endl;
        return;
    }

    cudaDeviceProp info;
    for(int i = 0; i < deviceCount; i++)
    {
        cudaGetDeviceProperties(&info, i);
        std::cout << "Device " << i << std::endl;
        std::cout << "    Name:                    " << std::string(info.name) << std::endl;
        std::cout << "    Glocbal memory:          " << info.totalGlobalMem/1024.0/1024.0 << " MB"<< std::endl;
        std::cout << "    Shared memory per block: " << info.sharedMemPerBlock/1024.0 << " KB"<< std::endl;
        std::cout << "    Warp size:               " << info.warpSize<< std::endl;
        std::cout << "    Max thread per block:    " << info.maxThreadsPerBlock<< std::endl;
        std::cout << "    Thread dimension limits: " << info.maxThreadsDim[0]<< " x "
                                                     << info.maxThreadsDim[1]<< " x "
                                                     << info.maxThreadsDim[2]<< std::endl;
        std::cout << "    Max grid size:           " << info.maxGridSize[0]<< " x "
                                                     << info.maxGridSize[1]<< " x "
                                                     << info.maxGridSize[2]<< std::endl;
        std::cout << "    Compute capability:      " << info.major << "." << info.minor << std::endl;
    }
}

void generateBGRA8K(uint8_t *buffer, int dataSize)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> sampler(0, 255);

    for(int i = 0; i < dataSize / 4; i++)
    {
        buffer[i * 4] = sampler(gen);
        buffer[i * 4 + 1] = sampler(gen);
        buffer[i * 4 + 2] = sampler(gen);
        buffer[i * 4 + 3] = 255;
    }
}

void convertPixelFormatCPU(uint8_t *inputBGRA, uint8_t *outputYUV, int numPixels)
{
    short3 YUV16;
    char3 YUV8;

    for(int idx = 0; idx < numPixels; idx++)
    {
        YUV16.x = 66 * inputBGRA[idx * 4 + 2] + 129 * inputBGRA[idx * 4 + 1] + 25 * inputBGRA[idx * 4];
        YUV16.y = -38 * inputBGRA[idx * 4 + 2] + -74 * inputBGRA[idx * 4 + 1] + 112 * inputBGRA[idx * 4];
        YUV16.z = 112 * inputBGRA[idx * 4 + 2] + -94 * inputBGRA[idx * 4 + 1] + -18 * inputBGRA[idx * 4];

        YUV8.x = (YUV16.x >> 8) + 16;
        YUV8.y = (YUV16.y >> 8) + 128;
        YUV8.z = (YUV16.z >> 8) + 128;

        (*reinterpret_cast<char3 *>(&outputYUV[idx * 3])) = YUV8;
    }
}

__global__ void convertPixelFormatCUDA(uint8_t *inputBGRA, uint8_t *outputYUV, int numPixels)
{
    int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    short3 YUV16;
    char3 YUV8;

    while(idx <= numPixels)
    {
        if(idx < numPixels)
        {
            YUV16.x = 66 * inputBGRA[idx * 4 + 2] + 129 * inputBGRA[idx * 4 + 1] + 25 * inputBGRA[idx * 4];
            YUV16.y = -38 * inputBGRA[idx * 4 + 2] + -74 * inputBGRA[idx * 4 + 1] + 112 * inputBGRA[idx * 4];
            YUV16.z = 112 * inputBGRA[idx * 4 + 2] + -94 * inputBGRA[idx * 4 + 1] + -18 * inputBGRA[idx * 4];

            YUV8.x = (YUV16.x >> 8) + 16;
            YUV8.y = (YUV16.y >> 8) + 128;
            YUV8.z = (YUV16.z >> 8) + 128;

            (*reinterpret_cast<char3 *>(&outputYUV[idx * 3])) = YUV8;
        }

        idx += stride;
    }
}
