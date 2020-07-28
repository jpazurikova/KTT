// This kernel code based on CUDPP.  Please see the notice in
// LICENSE_CUDPP.txt.

#if SORT_VECTOR == 2
#define SORTVECTYPE uint2
#elif SORT_VECTOR == 4
#define SORTVECTYPE uint4
#elif SORT_VECTOR == 8
#define SORTVECTYPE uint8
#endif

#if SCAN_VECTOR == 2
#define SCANVECTYPE uint2
#elif SCAN_VECTOR == 4
#define SCANVECTYPE uint4
#elif SCAN_VECTOR == 8
#define SCANVECTYPE uint8
#endif

uint scanLSB(const uint val, __local uint* s_data)
{
    // Set first half of shared mem to 0's
    int idx = get_local_id(0);
    s_data[idx] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Set 2nd half to thread local sum
    idx += get_local_size(0);

    // scan in local memory

    // Some of these __sync's are unnecessary due to warp synchronous
    // execution.  Right now these are left in to be consistent with
    // opencl version, since that has to execute on platforms where
    // thread groups are not synchronous (i.e. CPUs)
    uint t;
    s_data[idx] = val;     barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = 1; i < SORT_BLOCK_SIZE; i*=2) {
      t = s_data[idx - i];  barrier(CLK_LOCAL_MEM_FENCE);;
      s_data[idx] += t;      barrier(CLK_LOCAL_MEM_FENCE);;
    }
    return s_data[idx] - val;  // convert inclusive -> exclusive
}

SORTVECTYPE scan4(SORTVECTYPE idata, __local uint* ptr)
{
    SORTVECTYPE val4 = idata;
    SORTVECTYPE sum;

    // Scan the elements in idata within this thread
#if SORT_VECTOR == 2
    sum.x = val4.x;
    uint val = val4.y + sum.x;
#elif SORT_VECTOR == 4
    sum.x = val4.x;
    sum.y = val4.y + sum.x;
    sum.z = val4.z + sum.y;
    uint val = val4.w + sum.z;
#elif SORT_VECTOR == 8
    sum.s0 = val4.s0;
    sum.s1 = val4.s1 + sum.s0;
    sum.s2 = val4.s2 + sum.s1;
    sum.s3 = val4.s3 + sum.s2;
    sum.s4 = val4.s4 + sum.s3;
    sum.s5 = val4.s5 + sum.s4;
    sum.s6 = val4.s6 + sum.s5;
    uint val = val4.s7 + sum.s6;
#endif

    // Now scan those sums across the local work group
    val = scanLSB(val, ptr);

#if SORT_VECTOR == 2
    val4.x = val;
    val4.y = val + sum.x;
#elif SORT_VECTOR == 4
    val4.x = val;
    val4.y = val + sum.x;
    val4.z = val + sum.y;
    val4.w = val + sum.z;
#elif SORT_VECTOR == 8
    val4.s0 = val;
    val4.s1 = val + sum.s0;
    val4.s2 = val + sum.s1;
    val4.s3 = val + sum.s2;
    val4.s4 = val + sum.s3;
    val4.s5 = val + sum.s4;
    val4.s6 = val + sum.s5;
    val4.s7 = val + sum.s6;
#endif

    return val4;
}

//----------------------------------------------------------------------------
//
// radixSortBlocks sorts all blocks of data independently in shared
// memory.  Each thread block (CTA) sorts one block of SORT_VECTOR*CTA_SIZE elements
//
// The radix sort is done in two stages.  This stage calls radixSortBlock
// on each block independently, sorting on the basis of bits
// (startbit) -> (startbit + nbits)
//----------------------------------------------------------------------------

__kernel void radixSortBlocks(__global SORTVECTYPE* keysOut,
                              __global SORTVECTYPE* valuesOut,
                              __global SORTVECTYPE* keysIn,  
                              __global SORTVECTYPE* valuesIn)
{
    __local uint sMem[SORT_VECTOR*SORT_BLOCK_SIZE];

    // Get Indexing information
    const uint i = get_global_id(0);
    const uint tid = get_local_id(0);
    const uint localSize = get_local_size(0);

    // Load keys and vals from global memory
    SORTVECTYPE key, value;
    key = keysIn[i];
    value = valuesIn[i];

    if (i == 0) printf("beginning\n");
    if (i < 10) {
      printf("%d  %d  %d\n", i, key.x, value.x);
    }


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
        key.s0 = key.s0 + 2;
        key.s1 = key.s1 + 2;
        key.s2 = key.s2 + 2;
        key.s3 = key.s3 + 2;
        key.s4 = key.s4 + 2;
        key.s5 = key.s5 + 2;
        key.s6 = key.s6 + 2;
        key.s7 = key.s7 + 2;
#endif
        barrier(CLK_LOCAL_MEM_FENCE);;

       // Read keys out of local mem into registers, in prep for
        // write out to global mem
#if SORT_VECTOR == 2
        value.x = value.x + 2;
        value.y = value.y + 2;
#elif SORT_VECTOR == 4
        value.x = value.x + 2;
        value.y = value.y + 2;
        value.z = value.z + 2;
        value.w = value.w + 2;
#elif SORT_VECTOR == 8
        value.s0 = value.s0 + 2;
        value.s1 = value.s1 + 2;
        value.s2 = value.s2 + 2;
        value.s3 = value.s3 + 2;
        value.s4 = value.s4 + 2;
        value.s5 = value.s5 + 2;
        value.s6 = value.s6 + 2;
        value.s7 = value.s7 + 2;
#endif
        barrier(CLK_LOCAL_MEM_FENCE);;
    keysOut[i]   = key;
    valuesOut[i] = value;
    if (i == 0) printf("end\n");
    if (i < 10) {
      printf("%d  %d  %d\n", i, key.x, value.x);
    }
}
