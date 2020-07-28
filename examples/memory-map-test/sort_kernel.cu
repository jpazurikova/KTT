// This kernel code based on CUDPP.  Please see the notice in
// LICENSE_CUDPP.txt.

typedef unsigned int uint;

typedef struct __builtin_align__(16) {
    uint4 a;
    uint4 b;
  } my_uint8;

#if SORT_VECTOR == 2
#define SORTVECTYPE uint2
#elif SORT_VECTOR == 4
#define SORTVECTYPE uint4
#elif SORT_VECTOR == 8
#define SORTVECTYPE my_uint8
#endif

#if SCAN_VECTOR == 2
#define SCANVECTYPE uint2
#elif SCAN_VECTOR == 4
#define SCANVECTYPE uint4
#elif SCAN_VECTOR == 8
#define SCANVECTYPE my_uint8
#endif

//----------------------------------------------------------------------------
//
// radixSortBlocks sorts all blocks of data independently in shared
// memory.  Each thread block (CTA) sorts one block of SORT_VECTOR*CTA_SIZE elements
//
// The radix sort is done in two stages.  This stage calls radixSortBlock
// on each block independently, sorting on the basis of bits
// (startbit) -> (startbit + nbits)
//----------------------------------------------------------------------------

extern "C" __global__ void radixSortBlocks(SORTVECTYPE* keysOut, SORTVECTYPE* valuesOut,
                              SORTVECTYPE* keysIn,  SORTVECTYPE* valuesIn)
{
    // Get Indexing information
    const uint i = threadIdx.x + (blockIdx.x * blockDim.x);
    const uint tid = threadIdx.x;
    const uint localSize = blockDim.x;

    // Load keys and vals from global memory
    SORTVECTYPE key, value;
    key = keysIn[i];
    value = valuesIn[i];
    
    if (i == 0) printf("beginning\n");
    if (i < 10) {
      printf("%d  %d  %d\n", i, key.x, value.x);
    }


    // For each of the 4 bits
    {

        // Read keys out of local mem into registers, in prep for
        // write out to global mem
#if SORT_VECTOR == 2
        key.x = key.x + 2;
        key.y = key.y + 2;
#elif SORT_VECTOR == 4
        key.x = key.x + 2;
        key.y = key.y + 2;
        key.z = key.z + 2;
        key.w = key.w + 2;
#elif SORT_VECTOR == 8
        key.a.x = key.a.x + 2;
        key.a.y = key.a.y + 2;
        key.a.z = key.a.z + 2;
        key.a.w = key.a.w + 2;
        key.b.x = key.b.x + 2;
        key.b.y = key.b.y + 2;
        key.b.z = key.b.z + 2;
        key.b.w = key.b.w + 2;
#endif
        __syncthreads();

    keysOut[i]   = key;
    valuesOut[i] = value;
    if (i == 0) printf("end\n");
    if (i < 10) {
      printf("%d  %d  %d\n", i, key.x, value.x);
    }
    }
}
