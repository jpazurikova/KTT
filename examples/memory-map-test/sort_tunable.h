#pragma once

#include "tuner_api.h"
static const int SORT_BITS = 32;
typedef unsigned int uint;

static uint nbits = 4;

class TunableSort : public ktt::TuningManipulator {
  public:
    // Constructor takes ids of kernel arguments that will be updated or added
    TunableSort(
        const std::vector<ktt::KernelId>& kernelIds,
        const int size,
        const ktt::ArgumentId keysOutId,
        const ktt::ArgumentId valuesOutId,
        const ktt::ArgumentId keysInId,
        const ktt::ArgumentId valuesInId):
      kernelIds(kernelIds),
      size(size),
      keysOutId(keysOutId),
      valuesOutId(valuesOutId),
      keysInId(keysInId),
      valuesInId(valuesInId)
    {
    }

    // Run the code with kernels
    void launchComputation(const ktt::KernelId) override {

      std::vector<ktt::ParameterPair> parameterValues = getCurrentConfiguration();

      int sortBlockSize = (int)getParameterValue("SORT_BLOCK_SIZE", parameterValues);
      int sortVectorSize = (int)getParameterValue("SORT_VECTOR", parameterValues);
      const ktt::DimensionVector workGroupDimensionsSort(sortBlockSize, 1, 1);
      const ktt::DimensionVector ndRangeDimensionsSort(size/sortVectorSize, 1, 1);

      int scanBlockSize = (int)getParameterValue("SCAN_BLOCK_SIZE", parameterValues);
      int scanVectorSize = (int)getParameterValue("SCAN_VECTOR", parameterValues);
      const ktt::DimensionVector workGroupDimensionsScan(scanBlockSize, 1, 1);
      const ktt::DimensionVector ndRangeDimensionsScan(size/scanVectorSize, 1, 1);
      bool swap = true;

        //radixSortBlocks
        //  <<<radixBlocks, SORT_BLOCK_SIZE, 4 * sizeof(uint)*SORT_BLOCK_SIZE>>>
        //  (nbits, startbit, tempKeys, tempValues, keys, values);
        runKernel(kernelIds[0], ndRangeDimensionsSort, workGroupDimensionsSort);
    }

  private:
    std::vector<ktt::KernelId> kernelIds; // Ids of the internal kernels
    int size;
    ktt::ArgumentId keysOutId;
    ktt::ArgumentId valuesOutId;
    ktt::ArgumentId keysInId;
    ktt::ArgumentId valuesInId;
};
