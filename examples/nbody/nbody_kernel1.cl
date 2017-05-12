#if USE_CONSTANT_MEMORY == 0
    #define MEMORY_TYPE_AOS __global  const
    #define MEMORY_TYPE_SOA __global  const
#elif USE_CONSTANT_MEMORY == 1
    #if USE_SOA == 0
        #define MEMORY_TYPE_AOS __constant
        #define MEMORY_TYPE_SOA __global  const
    #elif USE_SOA > 0
        #define MEMORY_TYPE_AOS __global  const
        #define MEMORY_TYPE_SOA __constant
    #endif // USE_SOA
#endif // USE_CONSTANT_MEMORY

#if VECTOR_TYPE == 1
    // typedef float vector; // unsupported yet
#elif VECTOR_TYPE == 2
    // typedef float2 vector; // unsupported yet
#elif VECTOR_TYPE == 4
    typedef float4 vector;
#elif VECTOR_TYPE == 8
    // typedef float8 vector; // unsupported yet
#elif VECTOR_TYPE == 16
    // typedef float16 vector; // unsupported yet
#endif // VECTOR_TYPE

// method to calculate acceleration caused by body J
float3 computeAcc(float posI[3], // position of body I
	float posJX, float posJY, float posJZ, float bJMass, // position and mass of body J
	float softeningSqr) // to avoid infinities and zero division
{
	float3 acc, d;

    d.x = posJX - posI[0];
    d.y = posJY - posI[1];
    d.z = posJZ - posI[2];

    float distSqr = (d.x * d.x) + (d.y * d.y) + (d.z * d.z) + softeningSqr;
    float invDist = rsqrt(distSqr);
    float f = bJMass * invDist * invDist * invDist;

    acc.x = d.x * f;
    acc.y = d.y * f;
    acc.z = d.z * f;

    return acc;
}

// method to load thread specific data from global memory
void loadThreadData(
	MEMORY_TYPE_AOS vector* oldBodyInfo, MEMORY_TYPE_AOS vector* oldVel, // global data
	float bodyPos[3], float bodyVel[3], float bodyAcc[3], float* bodyMass, // thread data
	int start, int end) // indices
{
	int tid = get_local_id(0);
    int length = end - start + 1;
	
	if (tid < length) {
        // store 'thread specific' body info to registers
        bodyPos[0] = oldBodyInfo[start + tid].x;
        bodyPos[1] = oldBodyInfo[start + tid].y;
        bodyPos[2] = oldBodyInfo[start + tid].z;
		
        bodyVel[0] = oldVel[start + tid].x;
        bodyVel[1] = oldVel[start + tid].y;
        bodyVel[2] = oldVel[start + tid].z;
		
		*bodyMass = oldBodyInfo[start + tid].w;
		// erase acceleration buffer
		bodyAcc[0] = bodyAcc[1] = bodyAcc[2] = 0.f;
    }
}

// method to process complete block, i.e. part of the bodies array where
// each body's acceleration is added to result
void processCompleteBlock(
	MEMORY_TYPE_AOS vector* oldBodyInfo, // global data
	float bufferPosX[WORK_GROUP_SIZE_X], // buffers
	float bufferPosY[WORK_GROUP_SIZE_X],
	float bufferPosZ[WORK_GROUP_SIZE_X],
	float bufferMass[WORK_GROUP_SIZE_X],
	float bodyAcc[3], // thread specific data
	float bodyPos[3], 
	float softeningSqr, // used by acceleration
	int start) // initial index, included
{
    int tid = get_local_id(0);
    // load new values to buffer.
	// We know that all threads can be used now, so no condition is necessary
	bufferPosX[tid] = oldBodyInfo[start + tid].x;
	bufferPosY[tid] = oldBodyInfo[start + tid].y;
	bufferPosZ[tid] = oldBodyInfo[start + tid].z;
	bufferMass[tid] = oldBodyInfo[start + tid].w;
    barrier(CLK_LOCAL_MEM_FENCE);

    // calculate the acceleration between the thread body and each other body loaded to buffer
    # pragma unroll INNER_UNROLL_FACTOR1
    for(int i =  0; i < WORK_GROUP_SIZE_X; i++) {
        int index = i;
		float3 acc = computeAcc(bodyPos,
			bufferPosX[index], bufferPosY[index], bufferPosZ[index], bufferMass[index],
			softeningSqr);
		bodyAcc[0] += acc.x;
		bodyAcc[1] += acc.y;
		bodyAcc[2] += acc.z;
    }
    barrier(CLK_LOCAL_MEM_FENCE); // sync threads
}

// method to process final block, i.e. part of the molecule array where the algorithm terminates
void processFinalBlock(
	MEMORY_TYPE_AOS vector* oldBodyInfo, // global data
	float bufferPosX[WORK_GROUP_SIZE_X], // buffers
	float bufferPosY[WORK_GROUP_SIZE_X],
	float bufferPosZ[WORK_GROUP_SIZE_X],
	float bufferMass[WORK_GROUP_SIZE_X],
	float bodyAcc[3], // thread specific data
	float bodyPos[3], 
	float softeningSqr, // used by acceleration
	int start, int end) // initial (included) / end index
{
    int tid = get_local_id(0);
    int length = end - start + 1;
    int topIndex = length;
    if (length < 0) {
        return; 
    } // continue just with threads that won't access wrong memory
	
    // load new values to buffer
    if (tid < topIndex) {
		bufferPosX[tid] = oldBodyInfo[start + tid].x;
		bufferPosY[tid] = oldBodyInfo[start + tid].y;
		bufferPosZ[tid] = oldBodyInfo[start + tid].z;
		bufferMass[tid] = oldBodyInfo[start + tid].w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // calculate the acceleration between the thread body and each other body loaded to buffer
    int count = length / INNER_UNROLL_FACTOR2;
    int tmp  = INNER_UNROLL_FACTOR2 * count;
    # pragma unroll INNER_UNROLL_FACTOR2
    for(int i =  0; i < tmp; i++) {
        int index = i;
		float3 acc = computeAcc(bodyPos, bufferPosX[index], bufferPosY[index], bufferPosZ[index], bufferMass[index], softeningSqr);
		bodyAcc[0] += acc.x;
		bodyAcc[1] += acc.y;
		bodyAcc[2] += acc.z;
    }
	// finish those not processed in the block above, if any
    for(int i =  tmp; i < length; i++) {
        int index = i;
		float3 acc = computeAcc(bodyPos, bufferPosX[index], bufferPosY[index], bufferPosZ[index], bufferMass[index], softeningSqr);
		bodyAcc[0] += acc.x;
		bodyAcc[1] += acc.y;
		bodyAcc[2] += acc.z;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
	
// kernel calculating new position and velocity for n-bodies
__kernel void nbody_kernel(float timeDelta,
	MEMORY_TYPE_AOS vector* oldBodyInfo, // pos XYZ, mass
	__global vector* newBodyInfo,
	MEMORY_TYPE_AOS vector* oldVel, // XYZ, W unused, will be set to 0.f
	__global vector* newVel, // XYZ, W set to 0.f
	float damping, 
	float softeningSqr)
{
	// indices
	int n = get_global_size(0);
	int gtid = get_global_id(0);
	int blocks = n / WORK_GROUP_SIZE_X;
		
	// buffers for bodies info processed by the work group
	__local float bufferPosX[WORK_GROUP_SIZE_X];
	__local float bufferPosY[WORK_GROUP_SIZE_X];
	__local float bufferPosZ[WORK_GROUP_SIZE_X];
	__local float bufferMass[WORK_GROUP_SIZE_X];
	
    // each thread holds a position/mass of the body it represents
    float bodyPos[3];
    float bodyVel[3];
	float bodyAcc[3];
	float bodyMass;

	// load data
	loadThreadData(oldBodyInfo, oldVel, bodyPos, bodyVel, bodyAcc, &bodyMass,
		get_group_id(0) * WORK_GROUP_SIZE_X, // start index
		min(WORK_GROUP_SIZE_X * ((int)get_group_id(0) + 1) - 1, n - 1)); // end index

	// start the calculation, process whole blocks
    for (int i = 0; i < blocks; i++) {
        processCompleteBlock(
			oldBodyInfo,
			bufferPosX, bufferPosY, bufferPosZ, bufferMass,
			bodyAcc, bodyPos, 
			softeningSqr, 
			i * WORK_GROUP_SIZE_X); // start index is the first body in the block being processed
    }
    // at the end, do the final block
    processFinalBlock(
		oldBodyInfo,
		bufferPosX, bufferPosY, bufferPosZ, bufferMass,
		bodyAcc, bodyPos, 
		softeningSqr,
		blocks * WORK_GROUP_SIZE_X, n-1);

	// 'export' result
	if (gtid < n) {
		// calculate resulting position 	
		float resPosX = bodyPos[0] + timeDelta * bodyVel[0] + damping * timeDelta * timeDelta * bodyAcc[0];
		float resPosY = bodyPos[1] + timeDelta * bodyVel[1] + damping * timeDelta * timeDelta * bodyAcc[1];
		float resPosZ = bodyPos[2] + timeDelta * bodyVel[2] + damping * timeDelta * timeDelta * bodyAcc[2];
		newBodyInfo[gtid] = (vector)(resPosX, resPosY, resPosZ, bodyMass);
		// calculate resulting velocity	
		float resVelX = bodyVel[0] + timeDelta * bodyAcc[0];
		float resVelY = bodyVel[1] + timeDelta * bodyAcc[1];
		float resVelZ = bodyVel[2] + timeDelta * bodyAcc[2];
		newVel[gtid] = (vector)(resVelX, resVelY, resVelZ, 0.f);
	}
}