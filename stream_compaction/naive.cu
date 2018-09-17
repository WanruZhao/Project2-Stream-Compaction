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
    }
}
