#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <stdlib.h>
#include <limits.h>

#include "tuner_api.h"
#include "sort_reference.h"
#include "sort_tunable.h"

#ifndef RAND_MAX
#define RAND_MAX UINT_MAX
#endif

#if defined(_MSC_VER)
    const std::string kernelFilePrefix = "";
#else
    const std::string kernelFilePrefix = "../";
#endif

#if KTT_CUDA_EXAMPLE
    const std::string defaultKernelFile = kernelFilePrefix + "../examples/memory-map-test/sort_kernel.cu";
    const auto computeAPI = ktt::ComputeAPI::CUDA;
#elif KTT_OPENCL_EXAMPLE
    const std::string defaultKernelFile = kernelFilePrefix + "../examples/memory-map-test/sort_kernel.cl";
    const auto computeAPI = ktt::ComputeAPI::OpenCL;
#endif

int main(int argc, char** argv)
{
  // Initialize platform and device index
  ktt::PlatformIndex platformIndex = 0;
  ktt::DeviceIndex deviceIndex = 0;
  std::string kernelFile = defaultKernelFile;
  int problemSize = 32; // In MiB

  if (argc >= 2)
  {
    platformIndex = std::stoul(std::string{argv[1]});
    if (argc >= 3)
    {
      deviceIndex = std::stoul(std::string{argv[2]});
      if (argc >= 4)
      {
        problemSize = atoi(argv[3]);
      }
    }
  }


  if (argc >= 5)
  {
    kernelFile = std::string{argv[4]};
  }
  
  int size = problemSize * 1024 * 1024 / sizeof(unsigned int);

  // Create input and output vectors and initialize with pseudorandom numbers
  std::vector<unsigned int> keysIn(size);
  std::vector<unsigned int> keysOut(size);
  std::vector<unsigned int> valuesIn(size);
  std::vector<unsigned int> valuesOut(size);

  //srand((unsigned int)time(NULL));
  srand(123);
  for (int i = 0; i < size; i++)
  {
    valuesIn[i] = keysIn[i] = rand(); //i%1024;//rand();
  }
  for (int i = 0; i < 10; i++)
    printf("%d %d %d\n", i, valuesIn[i], keysIn[i]);

  // Create tuner object for chosen platform and device
  ktt::Tuner tuner(platformIndex, deviceIndex, computeAPI);
  tuner.setGlobalSizeType(ktt::GlobalSizeType::OpenCL);
  tuner.setPrintingTimeUnit(ktt::TimeUnit::Microseconds);
  tuner.setLoggingLevel(ktt::LoggingLevel::Debug);

  // Declare kernels and their dimensions
  std::vector<ktt::KernelId> kernelIds(1);
  const ktt::DimensionVector ndRangeDimensions;
  const ktt::DimensionVector workGroupDimensions;

  kernelIds[0] = tuner.addKernelFromFile(kernelFile, std::string("radixSortBlocks"), ndRangeDimensions, workGroupDimensions);

  // Add arguments for kernels
  //all parameters with foo values (empty vectors or scalar 1) will be updated in tuning manipulator, as their value depends on tuning parameters
  
  ktt::ArgumentId keysOutId = tuner.addArgumentVector(keysOut, ktt::ArgumentAccessType::ReadWrite);
  ktt::ArgumentId valuesOutId = tuner.addArgumentVector(valuesOut, ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::HostZeroCopy, true);
  ktt::ArgumentId keysInId = tuner.addArgumentVector(keysIn, ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::HostZeroCopy, true);
  ktt::ArgumentId valuesInId = tuner.addArgumentVector(valuesIn, ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::HostZeroCopy, true);

  
  ktt::KernelId compositionId = tuner.addComposition("sort", kernelIds, std::make_unique<TunableSort>(kernelIds, size, keysOutId, valuesOutId, keysInId, valuesInId));

  //radixSortBlocks
  tuner.setCompositionKernelArguments(compositionId, kernelIds[0], std::vector<size_t>{keysOutId, valuesOutId, keysInId, valuesInId});

  tuner.addParameter(compositionId, "SORT_BLOCK_SIZE", {32, 64, 128, 256, 512, 1024});
  tuner.addParameter(compositionId, "SCAN_BLOCK_SIZE", {32, 64, 128, 256, 512, 1024});
  tuner.addParameter(compositionId, "SORT_VECTOR", {2,4,8});
  tuner.addParameter(compositionId, "SCAN_VECTOR", {2,4,8});
  auto workGroupConstraint = [](const std::vector<size_t>& vector) {return (float)vector.at(1)/vector.at(0) == (float)vector.at(2)/vector.at(3);};
  tuner.addConstraint(compositionId, {"SORT_BLOCK_SIZE", "SCAN_BLOCK_SIZE", "SORT_VECTOR", "SCAN_VECTOR"}, workGroupConstraint);

  tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.9);
  tuner.setReferenceClass(compositionId, std::make_unique<ReferenceSort>(valuesIn), std::vector<ktt::ArgumentId>{valuesOutId});

  std::vector<float> oneElement(1);
  ktt::OutputDescriptor output(valuesOutId, (void*)oneElement.data(), 1*sizeof(float));
  for (int i = 0; i < 10; i++) {
        tuner.tuneKernelByStep(compositionId, {output});
  }
  //tuner.tuneKernel(compositionId);
  tuner.printResult(compositionId, std::cout, ktt::PrintFormat::Verbose);
  tuner.printResult(compositionId, std::string("sort_result.csv"), ktt::PrintFormat::CSV);
  return 0;
}
