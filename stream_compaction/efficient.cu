#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernReduction(int n, int d, int *idata) {
			int index = blockDim.x * blockIdx.x + threadIdx.x;
			if (index >= n) return;
			
			int offset1 = 1 << d;
			int offset2 = 1 << (d + 1);

			idata[index * offset2 + offset2 - 1] += idata[index * offset2 + offset1 - 1];
		}

		__global__ void kernScan(int n, int d, int *idata) {
			int index = blockDim.x * blockIdx.x + threadIdx.x;
			if (index >= n) return;

			int offset1 = 1 << d;
			int offset2 = 1 << (d + 1);

			int leftIdx = index * offset2 + offset1 - 1;
			int rightIdx = index * offset2 + offset2 - 1;
			
			int originLeft = idata[leftIdx];
			idata[leftIdx] = idata[rightIdx];
			idata[rightIdx] += originLeft;
		}


		void scanCore(int nPow, int d, int *dev_idata) {
			dim3 gridDim;
			for (int i = 0; i < d; i++) {
				int scale = 1 << (i + 1);
				gridDim = ((nPow / scale + blockSize - 1) / blockSize);
				kernReduction << <gridDim, blockSize >> > (nPow / scale, i, dev_idata);
			}
			cudaMemset(dev_idata + nPow - 1, 0, sizeof(int));
			for (int i = d - 1; i >= 0; i--) {
				int scale = 1 << (i + 1);
				gridDim = ((nPow / scale + blockSize - 1) / blockSize);
				kernScan << <gridDim, blockSize >> > (nPow / scale, i, dev_idata);
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			int d = ilog2ceil(n);
			int nPow = 1 << d;
			int *dev_idata;
			cudaMalloc((void**)&dev_idata, nPow * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_idata failed!");
			cudaMemset(dev_idata + n, 0, (nPow - n) * sizeof(int));
			checkCUDAError("cudaMemset dev_idata failed!");

            timer().startGpuTimer();
            // TODO
			scanCore(nPow, d, dev_idata);

            timer().endGpuTimer();

			cudaMemcpy(odata , dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_idata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {

			int *dev_idata, *dev_bools, *dev_odata, *dev_indices;

			int d = ilog2ceil(n);
			int nPow = 1 << d;

			cudaMalloc((void**)&dev_idata, nPow * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_idata failed!");
			cudaMemset(dev_idata + n, 0, (nPow - n) * sizeof(int));
			checkCUDAError("cudaMemset dev_idata failed!");
			
			cudaMalloc((void**)&dev_bools, nPow * sizeof(int));
			checkCUDAError("cudaMalloc dev_bools failed!");
			cudaMalloc((void**)&dev_odata, nPow * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");
			cudaMalloc((void**)&dev_indices, nPow * sizeof(int));
			checkCUDAError("cudaMalloc dev_indices failed!");

            timer().startGpuTimer();
            // TODO

			dim3 gridDim((nPow + blockSize - 1) / blockSize);
			StreamCompaction::Common::kernMapToBoolean << <gridDim, blockSize >> > (nPow, dev_bools, dev_idata);			
			cudaMemcpy(dev_indices, dev_bools, nPow * sizeof(int), cudaMemcpyDeviceToDevice);
			scanCore(nPow, d, dev_indices);
			StreamCompaction::Common::kernScatter << <gridDim, blockSize >> > (nPow, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, nPow * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_bools);
			cudaFree(dev_indices);

			int count = 0;
			for (int i = 0; i < n; i++) {
				if (odata[i]) {
					count++;
				}
				else {
					break;
				}
			}
            return count;
        }
    }
}
