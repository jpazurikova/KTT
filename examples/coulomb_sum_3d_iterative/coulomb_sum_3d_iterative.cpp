#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "tuner_api.h"

class CoulombManipulator : public ktt::TuningManipulator
{
public:
    CoulombManipulator(const int atoms, const int gridSize, const float gridSpacing, const ktt::ArgumentId atomInfoPrecompId,
        const ktt::ArgumentId atomInfoZ2Id, const ktt::ArgumentId zIndexId, const std::vector<float>& atomInfoPrecomp,
        const std::vector<float>& atomInfoZ, const std::vector<float>& atomInfoZ2) :
        atoms(atoms),
        gridSize(gridSize),
        gridSpacing(gridSpacing),
        atomInfoPrecompId(atomInfoPrecompId),
        atomInfoZ2Id(atomInfoZ2Id),
        zIndexId(zIndexId),
        atomInfoPrecomp(atomInfoPrecomp),
        atomInfoZ(atomInfoZ),
        atomInfoZ2(atomInfoZ2)
    {}

    // LaunchComputation is responsible for actual execution of tuned kernel
    void launchComputation(const ktt::KernelId kernelId) override
    {
        // Get kernel data
        ktt::DimensionVector globalSize = getCurrentGlobalSize(kernelId);
        ktt::DimensionVector localSize = getCurrentLocalSize(kernelId);
        std::vector<ktt::ParameterPair> parameterValues = getCurrentConfiguration();

        globalSize.setSizeZ(1);

        // Iterate over slices
        for (int i = 0; i < gridSize; i++)
        {
            // Perform precomputation for 2D kernel
            float z = gridSpacing * float(i);
            if (getParameterValue("USE_SOA", parameterValues) == 0)
            {
                for (int j = 0; j < atoms; j++)
                {
                    atomInfoPrecomp[j * 4 + 2] = (z - atomInfoZ[j]) * (z - atomInfoZ[j]);
                }
                updateArgumentVector(atomInfoPrecompId, atomInfoPrecomp.data());
            }
            else
            {
                for (int j = 0; j < atoms; j++)
                {
                    atomInfoZ2[j] = (z - atomInfoZ[j]) * (z - atomInfoZ[j]);
                }
                updateArgumentVector(atomInfoZ2Id, atomInfoZ2.data());
            }
            updateArgumentScalar(zIndexId, &i);
        
            runKernel(kernelId, globalSize, localSize);
        }
    }

private:
    int atoms;
    int gridSize;
    float gridSpacing;
    ktt::ArgumentId atomInfoPrecompId;
    ktt::ArgumentId atomInfoZ2Id;
    ktt::ArgumentId zIndexId;
    std::vector<float> atomInfoPrecomp;
    std::vector<float> atomInfoZ;
    std::vector<float> atomInfoZ2;
};

int main(int argc, char** argv)
{
    // Initialize platform index, device index and paths to kernels
    size_t platformIndex = 0;
    size_t deviceIndex = 0;
    std::string kernelFile = "../examples/coulomb_sum_3d_iterative/coulomb_sum_3d_iterative_kernel.cl";
    std::string referenceKernelFile = "../examples/coulomb_sum_3d_iterative/coulomb_sum_3d_iterative_reference_kernel.cl";

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string(argv[1]));
        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string(argv[2]));
            if (argc >= 4)
            {
                kernelFile = std::string(argv[3]);
                if (argc >= 5)
                {
                    referenceKernelFile = std::string(argv[4]);
                }
            }
        }
    }

    // Set the problem size and declare data variables
    const int atoms = 4000;
    const int gridSize = 256;
    float gridSpacing = 0.5f;
    int zIndex = 0;
    std::vector<float> atomInfo;
    std::vector<float> atomInfoPrecomp;
    std::vector<float> atomInfoX;
    std::vector<float> atomInfoY;
    std::vector<float> atomInfoZ;
    std::vector<float> atomInfoZ2;
    std::vector<float> atomInfoW;
    std::vector<float> energyGrid;

    energyGrid.assign(gridSize * gridSize * gridSize, 0.0f);
    atomInfoX.resize(atoms);
    atomInfoY.resize(atoms);
    atomInfoZ.resize(atoms);
    atomInfoZ2.resize(atoms);
    atomInfoW.resize(atoms);
    atomInfo.resize(atoms * 4);
    atomInfoPrecomp.resize(atoms * 4);

    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(0.0f, 40.0f);
    for (int i = 0; i < atoms; i++)
    {
        atomInfoX.at(i) = distribution(engine);
        atomInfoY.at(i) = distribution(engine);
        atomInfoZ.at(i) = distribution(engine);
        atomInfoW.at(i) = distribution(engine)/40.0f;

        atomInfo.at((4 * i)) = atomInfoX.at(i);
        atomInfo.at((4 * i) + 1) = atomInfoY.at(i);
        atomInfo.at((4 * i) + 2) = atomInfoZ.at(i);
        atomInfo.at((4 * i) + 3) = atomInfoW.at(i);

        // do not store z, it will be rewritten anyway
        atomInfoPrecomp.at((4 * i)) = atomInfoX.at(i);
        atomInfoPrecomp.at((4 * i) + 1) = atomInfoY.at(i);
        atomInfoPrecomp.at((4 * i) + 3) = atomInfoW.at(i);
    }

    const ktt::DimensionVector ndRangeDimensions(gridSize, gridSize, gridSize);
    const ktt::DimensionVector workGroupDimensions;
    const ktt::DimensionVector referenceWorkGroupDimensions(16, 16);

    // Create tuner object for specified platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex);
    
    // Add two kernels to tuner, one of the kernels acts as reference kernel
    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "directCoulombSum", ndRangeDimensions, workGroupDimensions);
    ktt::KernelId referenceKernelId = tuner.addKernelFromFile(referenceKernelFile, "directCoulombSumReference", ndRangeDimensions, referenceWorkGroupDimensions);

    // Add all arguments utilized by kernels
    ktt::ArgumentId atomInfoId = tuner.addArgumentVector(atomInfo, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId atomInfoPrecompId = tuner.addArgumentVector(atomInfoPrecomp, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId atomInfoXId = tuner.addArgumentVector(atomInfoX, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId atomInfoYId = tuner.addArgumentVector(atomInfoY, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId atomInfoZId = tuner.addArgumentVector(atomInfoZ, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId atomInfoZ2Id = tuner.addArgumentVector(atomInfoZ2, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId atomInfoWId = tuner.addArgumentVector(atomInfoW, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId numberOfAtomsId = tuner.addArgumentScalar(atoms);
    ktt::ArgumentId gridSpacingId = tuner.addArgumentScalar(gridSpacing);
    ktt::ArgumentId zIndexId = tuner.addArgumentScalar(zIndex);
    ktt::ArgumentId energyGridId = tuner.addArgumentVector(energyGrid, ktt::ArgumentAccessType::ReadWrite);

    // Set kernel arguments for both tuned kernel and reference kernel, order of arguments is important
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{atomInfoPrecompId, atomInfoXId, atomInfoYId, atomInfoZ2Id, atomInfoWId,
        numberOfAtomsId, gridSpacingId, zIndexId, energyGridId});
    tuner.setKernelArguments(referenceKernelId, std::vector<ktt::ArgumentId>{atomInfoId, numberOfAtomsId, gridSpacingId, energyGridId});

    // Multiply workgroup size in dimensions x and y by two parameters that follow (effectively setting workgroup size to parameters' values)
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", std::vector<size_t>{4, 8, 16, 32}, ktt::ThreadModifierType::Local,
        ktt::ThreadModifierAction::Multiply, ktt::Dimension::X);
    tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Y", std::vector<size_t>{1, 2, 4, 8, 16, 32}, ktt::ThreadModifierType::Local,
        ktt::ThreadModifierAction::Multiply, ktt::Dimension::Y);
    tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR", std::vector<size_t>{0, 1, 2, 4, 8, 16, 32});
    tuner.addParameter(kernelId, "USE_CONSTANT_MEMORY", std::vector<size_t>{0, 1});
    tuner.addParameter(kernelId, "VECTOR_TYPE", std::vector<size_t>{1, 2, 4, 8});
    tuner.addParameter(kernelId, "USE_SOA", std::vector<size_t>{0, 1, 2});

    // Using vectorized SoA only makes sense when vectors are longer than 1
    auto vectorizedSoA = [](std::vector<size_t> vector) {return vector.at(0) > 1 || vector.at(1) != 2;};
    tuner.addConstraint(kernelId, vectorizedSoA, std::vector<std::string>{"VECTOR_TYPE", "USE_SOA"});
    // Ensure sufficient parallelism
    auto par = [](std::vector<size_t> vector) {return vector.at(0) * vector.at(1) >= 64;};
    tuner.addConstraint(kernelId, par, {"WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"});

    // Divide NDRange in dimension x by OUTER_UNROLL_FACTOR
    tuner.addParameter(kernelId, "OUTER_UNROLL_FACTOR", std::vector<size_t>{1, 2, 4, 8}, ktt::ThreadModifierType::Global,
        ktt::ThreadModifierAction::Divide, ktt::Dimension::X);
        
    // Specify custom tolerance threshold for validation of floating point arguments. Default threshold is 1e-4.
    tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.01);

    // Set reference kernel which validates results provided by tuned kernel, provide list of arguments which will be validated
    tuner.setReferenceKernel(kernelId, referenceKernelId, std::vector<ktt::ParameterPair>{}, std::vector<ktt::ArgumentId>{energyGridId});

    // Set tuning manipulator, which implements custom method for launching the kernel
    tuner.setTuningManipulator(kernelId, std::make_unique<CoulombManipulator>(atoms, gridSize, gridSpacing, atomInfoPrecompId, atomInfoZ2Id,
        zIndexId, atomInfoPrecomp, atomInfoZ, atomInfoZ2));
    
    // Launch kernel tuning
    tuner.tuneKernel(kernelId);

    // Print tuning results to standard output and to output.csv file
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, "coulomb_sum_3d_iterative_output.csv", ktt::PrintFormat::CSV);

    return 0;
}
