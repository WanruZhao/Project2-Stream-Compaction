#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScan(int n, int d, int *idata, int *odata) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n) {
				return;
			}
			int offset = 1 << (d - 1);
			if (index >= offset) {
				odata[index] = idata[index - offset] + idata[index];
			}
			else {
				odata[index] = idata[index];
			}
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			int *dev_idata, *dev_odata;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_idata failed!");

			int d = ilog2ceil(n);

            timer().startGpuTimer();
            // TODO
			dim3 gridDim((n + blockSize - 1) / blockSize);
			for (int i = 1; i <= d; i++) {
				kernNaiveScan << <gridDim, blockSize >> > (n, i, dev_idata, dev_odata);
				std::swap(dev_idata, dev_odata);
			}
			std::swap(dev_idata, dev_odata);

            timer().endGpuTimer();

			odata[0] = 0;
			cudaMemcpy(odata + 1, dev_odata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
			
			cudaFree(dev_idata);
			cudaFree(dev_odata);
		}
		

		__global__ void kernSharedNaiveScan(int n, int d, int *odata, int *auxidata, const int *idata, int isMulti) {
			extern __shared__ int temp[];

			int index = threadIdx.x;
			int boffset = blockIdx.x * blockDim.x;

			int pout = 1, pin = 0;
			temp[pout * n + index + boffset * 2] = idata[pout * n + index + boffset];
			
			__syncthreads();
			for(int i = 1; i < (1 << d); i *= 2) {
				pout = 1 - pout;
				pin = 1 - pout;

				if(index >= i) {
					temp[pout * n + index + boffset * 2] += temp[pin * n + index + boffset * 2 - i];
				} else {
					temp[pout * n + index + boffset * 2] = temp[pin * n + index + boffset * 2];
				}

				__syncthreads();
			}

			if(index == n - 1 && isMulti) {
				if(blockIdx.x)
					auxidata[blockIdx.x] = temp[pout * n + index + boffset];
				else
					auxidata[blockIdx.x] = 0;
			}

			odata[index + boffset] = temp[pout * n + index + boffset];	
		}

		__global__ void kernAddAuxi(int n, int *odata, const int *auxi) {
			extern __shared__ int temp[];

			int index = blockDim.x * blockIdx.x + threadIdx.x;
			if(index >= n) return;

			temp[blockIdx.x] = auxi[blockIdx.x];

			__syncthreads();

			odata[index] += temp[blockIdx.x];
		}



		void shared_scan(int n, int *odata, const int *idata)
		{
			int blockCount = (n + blockSize - 1) / blockSize;
			int lastBlockN = n - (blockCount - 1) * blockSize;
			int d = ilog2ceil(blockSize);

			int *dev_idata, *dev_odata, *dev_auxi, *dev_tauxi;

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");

			cudaMalloc((void**)&dev_auxi, blockCount * sizeof(int));
			checkCUDAError("cudaMalloc dev_auxi failed!");

			cudaMalloc((void**)&dev_tauxi, blockCount * sizeof(int));
			checkCUDAError("cudaMalloc dev_tauxi failed!");

			int blocknum = blockSize;
			int sharedMemory = 2 * blockSize * sizeof(int);
			kernSharedNaiveScan<<<blockCount, blocknum, sharedMemory>>>(blocknum, d, dev_odata, dev_auxi, dev_idata, 1);

			// assump blockSize <= blockSize
			sharedMemory = 2 * blockCount * sizeof(int);
			d = ilog2ceil(blockCount);
			kernSharedNaiveScan<<<1, blockCount, sharedMemory>>>(blocknum, d, dev_tauxi, NULL, dev_auxi, 0);

			sharedMemory = blockCount * sizeof(int);
			kernAddAuxi<<<blockCount, blocknum, sharedMemory>>> (blocknum, dev_odata, dev_tauxi);

			odata[0] = 0;
			cudaMemcpy(odata + 1, dev_odata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_auxi);

		}
    }
}
