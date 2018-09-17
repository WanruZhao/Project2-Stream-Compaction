#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {

            if(n == 0) return;

	        timer().startCpuTimer();
            // TODO

            odata[0] = 0;
            for(int i = 1; i < n; i++) {
                odata[i] = idata[i - 1] + odata[i - 1];
            }

	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            // TODO

            int count = 0;
            for(int i = 0; i < n; i++) {
                if(idata[i] != 0) {
                    odata[count++] = idata[i];
                }
            }

	        timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            if(n == 0) return 0;
	        timer().startCpuTimer();
            // TODO
            
            int count = 0;

            int *tmp = (int*)malloc(n * sizeof(int));
            for(int i = 0; i < n; i++) {
                if(idata[i] == 0) {
                    tmp[i] = 0;
                } else {
                    tmp[i] = 1;
                }
            }

            int *scan = (int*)malloc(n * sizeof(int));
            scan[0] = 0;
            for(int i = 1; i < n; i++) {
                scan[i] = tmp[i - 1] + scan[i - 1];
            }
            
            for(int i = 0; i < n; i++) {
                if(tmp[i]) {
                    odata[scan[i]] = idata[i];
                    count++;
                }
            }

	        timer().endCpuTimer();
			free(tmp);
			free(scan);
            return count;
        }
    }
}
