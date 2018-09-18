#include <cuda.h>
#include <cuda_runtime.h>
#include "radix.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernRadixEArray(int n, int p, int *bdata, int *edata, const int *idata) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if(index >= n) return;

            int digit = idata[index] & p;
            
            edata[index] = digit ? 0 : 1;
            bdata[index] = digit ? 1 : 0;
        }

        __global__ void kernRadixDArray(int n, int totalFalse, int *ddata, const int *fdata, int *bdata) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if(index >= n) return;

            int f = fdata[index];
            int t = index - f + totalFalse;
            ddata[index] = bdata[index] ? t : f;
			bdata[index] = 1;
        }

        void sort(int n, int *odata, const int *idata){

            int d = ilog2ceil(n);
            int nPow = 1 << d;

            int *dev_idataPow, *dev_bdataPow, *dev_edataPow, *dev_fdataPow, *dev_ddataPow, *dev_odataPow;
            int *en, *fn;

			en = (int*)std::malloc(sizeof(int));
			fn = (int*)std::malloc(sizeof(int));

            cudaMalloc((void**)&dev_idataPow, nPow * sizeof(int));
			checkCUDAError("cudaMalloc dev_idataPow failed!");
			cudaMemcpy(dev_idataPow, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_idataPow failed!");
			cudaMemset(dev_idataPow + n, 0, (nPow - n) * sizeof(int));
            checkCUDAError("cudaMemset dev_idataPow failed!");
            
            cudaMalloc((void**)&dev_edataPow, nPow * sizeof(int));
            checkCUDAError("cudaMalloc dev_edataPow failed!");
            cudaMemset(dev_edataPow, 0, nPow * sizeof(int));
            cudaMalloc((void**)&dev_bdataPow, nPow * sizeof(int));
            checkCUDAError("cudaMalloc dev_bdataPow failed!");
            cudaMemset(dev_bdataPow, 0, nPow * sizeof(int));
            cudaMalloc((void**)&dev_fdataPow, nPow * sizeof(int));
            checkCUDAError("cudaMalloc dev_fdataPow failed!");
            cudaMemset(dev_fdataPow, 0, nPow * sizeof(int));
            cudaMalloc((void**)&dev_ddataPow, nPow * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemset(dev_ddataPow, 0, nPow * sizeof(int));
            cudaMalloc((void**)&dev_odataPow, nPow * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMemset(dev_odataPow, 0, nPow * sizeof(int));
            
            timer().startGpuTimer();
            
            dim3 gridDim((n + blockSize - 1) / blockSize);
            
            for(int p = 1; p <= (1 << 6); p = p << 1) {
                kernRadixEArray<<<gridDim, blockSize>>>(n, p, dev_bdataPow, dev_edataPow, dev_idataPow);
                cudaMemcpy(dev_fdataPow, dev_edataPow, nPow * sizeof(int), cudaMemcpyDeviceToDevice);
                StreamCompaction::Efficient::scanCore(nPow, d, dev_fdataPow);
                cudaMemcpy(en, dev_edataPow + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(fn, dev_fdataPow + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                kernRadixDArray<<<gridDim, blockSize>>>(n, en[0] + fn[0], dev_ddataPow, dev_fdataPow, dev_bdataPow);
                StreamCompaction::Common::kernScatter<<<gridDim, blockSize>>>(n, dev_odataPow, dev_idataPow, dev_bdataPow, dev_ddataPow);
                std::swap(dev_idataPow, dev_odataPow);
            }
            std::swap(dev_idataPow, dev_odataPow);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odataPow, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idataPow);
            cudaFree(dev_bdataPow);
            cudaFree(dev_edataPow);
            cudaFree(dev_fdataPow);
            cudaFree(dev_ddataPow);
            cudaFree(dev_odataPow);

			free(en);
			free(fn);

        }

    }
}