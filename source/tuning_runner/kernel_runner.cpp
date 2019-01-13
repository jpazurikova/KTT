#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuning_runner/kernel_runner.h>
#include <utility/ktt_utility.h>
#include <utility/logger.h>
#include <utility/timer.h>

namespace ktt
{

KernelRunner::KernelRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, ComputeEngine* computeEngine) :
    argumentManager(argumentManager),
    kernelManager(kernelManager),
    computeEngine(computeEngine),
    resultValidator(argumentManager, this),
    manipulatorInterfaceImplementation(std::make_unique<ManipulatorInterfaceImplementation>(computeEngine)),
    kernelProfilingFlag(false)
{}

KernelResult KernelRunner::runKernel(const KernelId id, const KernelRunMode mode, const KernelConfiguration& configuration,
    const std::vector<OutputDescriptor>& output)
{
    if (!kernelManager->isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    const Kernel& kernel = kernelManager->getKernel(id);
    if (!resultValidator.hasReferenceResult(id))
    {
        resultValidator.computeReferenceResult(kernel, mode);
    }

    std::stringstream stream;
    stream << "Running kernel " << kernel.getName() << " with configuration: " << configuration;
    Logger::getLogger().log(LoggingLevel::Info, stream.str());

    KernelResult result;
    try
    {
        if (kernel.hasTuningManipulator())
        {
            auto manipulatorPointer = tuningManipulators.find(id);
            result = runKernelWithManipulator(kernel, manipulatorPointer->second.get(), configuration, output);
        }
        else
        {
            result = runKernelSimple(kernel, configuration, output);
        }
        validateResult(kernel, result, mode);
    }
    catch (const std::runtime_error& error)
    {
        computeEngine->synchronizeDevice();
        computeEngine->clearEvents();
        Logger::getLogger().log(LoggingLevel::Warning, std::string("Kernel run failed, reason: ") + error.what());
        result = KernelResult(kernel.getName(), configuration, error.what());
    }

    return result;
}

KernelResult KernelRunner::runKernel(const KernelId id, const KernelRunMode mode, const std::vector<ParameterPair>& configuration,
    const std::vector<OutputDescriptor>& output)
{
    const KernelConfiguration launchConfiguration = kernelManager->getKernelConfiguration(id, configuration);
    return runKernel(id, mode, launchConfiguration, output);
}

KernelResult KernelRunner::runComposition(const KernelId id, const KernelRunMode mode, const KernelConfiguration& configuration,
    const std::vector<OutputDescriptor>& output)
{
    if (!kernelManager->isComposition(id))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(id));
    }

    const KernelComposition& composition = kernelManager->getKernelComposition(id);
    const Kernel compatibilityKernel = composition.transformToKernel();
    if (!resultValidator.hasReferenceResult(id))
    {
        resultValidator.computeReferenceResult(compatibilityKernel, mode);
    }

    std::stringstream stream;
    stream << "Running kernel composition " << composition.getName() << " with configuration: " << configuration;
    Logger::getLogger().log(LoggingLevel::Info, stream.str());

    KernelResult result;
    try
    {
        auto manipulatorPointer = tuningManipulators.find(id);
        result = runCompositionWithManipulator(composition, manipulatorPointer->second.get(), configuration, output);
        validateResult(compatibilityKernel, result, mode);
    }
    catch (const std::runtime_error& error)
    {
        computeEngine->synchronizeDevice();
        computeEngine->clearEvents();
        Logger::getLogger().log(LoggingLevel::Warning, std::string("Kernel composition run failed, reason: ") + error.what());
        result = KernelResult(composition.getName(), configuration, error.what());
    }

    return result;
}

KernelResult KernelRunner::runComposition(const KernelId id, const KernelRunMode mode, const std::vector<ParameterPair>& configuration,
    const std::vector<OutputDescriptor>& output)
{
    const KernelConfiguration launchConfiguration = kernelManager->getKernelCompositionConfiguration(id, configuration);
    return runComposition(id, mode, launchConfiguration, output);
}

void KernelRunner::setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator)
{
    if (tuningManipulators.find(id) != tuningManipulators.end())
    {
        tuningManipulators.erase(id);
    }
    tuningManipulators.insert(std::make_pair(id, std::move(manipulator)));
}

void KernelRunner::setTuningManipulatorSynchronization(const KernelId id, const bool flag)
{
    if (flag && disabledSynchronizationManipulators.find(id) != disabledSynchronizationManipulators.end())
    {
        disabledSynchronizationManipulators.erase(id);
    }
    else if (!flag && disabledSynchronizationManipulators.find(id) == disabledSynchronizationManipulators.end())
    {
        disabledSynchronizationManipulators.insert(id);
    }
}

void KernelRunner::setKernelProfiling(const bool flag)
{
    kernelProfilingFlag = flag;
    manipulatorInterfaceImplementation->setKernelProfiling(flag);
}

void KernelRunner::setValidationMethod(const ValidationMethod method, const double toleranceThreshold)
{
    resultValidator.setValidationMethod(method);
    resultValidator.setToleranceThreshold(toleranceThreshold);
}

void KernelRunner::setValidationMode(const ValidationMode mode)
{
    resultValidator.setValidationMode(mode);
}

void KernelRunner::setValidationRange(const ArgumentId id, const size_t range)
{
    resultValidator.setValidationRange(id, range);
}

void KernelRunner::setArgumentComparator(const ArgumentId id, const std::function<bool(const void*, const void*)>& comparator)
{
    resultValidator.setArgumentComparator(id, comparator);
}

void KernelRunner::setReferenceKernel(const KernelId id, const KernelId referenceId, const std::vector<ParameterPair>& referenceConfiguration,
    const std::vector<ArgumentId>& validatedArgumentIds)
{
    resultValidator.setReferenceKernel(id, referenceId, referenceConfiguration, validatedArgumentIds);
}

void KernelRunner::setReferenceClass(const KernelId id, std::unique_ptr<ReferenceClass> referenceClass,
    const std::vector<ArgumentId>& validatedArgumentIds)
{
    resultValidator.setReferenceClass(id, std::move(referenceClass), validatedArgumentIds);
}

void KernelRunner::clearReferenceResult(const KernelId id)
{
    resultValidator.clearReferenceResults(id);
}

KernelArgument KernelRunner::downloadArgument(const ArgumentId id) const
{
    return computeEngine->downloadArgumentObject(id, nullptr);
}

void KernelRunner::clearBuffers(const ArgumentAccessType accessType)
{
    computeEngine->clearBuffers(accessType);
}

void KernelRunner::clearBuffers()
{
    computeEngine->clearBuffers();
}

void KernelRunner::setPersistentArgumentUsage(const bool flag)
{
    computeEngine->setPersistentBufferUsage(flag);
}

KernelResult KernelRunner::runKernelSimple(const Kernel& kernel, const KernelConfiguration& configuration,
    const std::vector<OutputDescriptor>& output)
{
    KernelId kernelId = kernel.getId();
    std::string kernelName = kernel.getName();
    std::string source = kernelManager->getKernelSourceWithDefines(kernelId, configuration);
    KernelRuntimeData kernelData(kernelId, kernelName, source, configuration.getGlobalSize(), configuration.getLocalSize(), kernel.getArgumentIds(),
        configuration.getLocalMemoryModifiers());

    KernelResult result;
    if (kernelProfilingFlag)
    {
        EventId id = computeEngine->runKernelWithProfiling(kernelData, argumentManager->getArguments(kernel.getArgumentIds()),
            computeEngine->getDefaultQueue());
        while (computeEngine->getRemainingKernelProfilingRuns(kernelData.getName(), kernelData.getSource()) > 0)
        {
            if (computeEngine->getRemainingKernelProfilingRuns(kernelData.getName(), kernelData.getSource()) > 1)
            {
                computeEngine->clearBuffers(ArgumentAccessType::ReadWrite);
                computeEngine->clearBuffers(ArgumentAccessType::WriteOnly);
            }
            computeEngine->runKernelWithProfiling(kernelData, argumentManager->getArguments(kernel.getArgumentIds()),
                computeEngine->getDefaultQueue());
        }
        result = computeEngine->getKernelResultWithProfiling(id, output);
    }
    else
    {
        result = computeEngine->runKernel(kernelData, argumentManager->getArguments(kernel.getArgumentIds()), output);
    }

    result.setConfiguration(configuration);
    return result;
}

KernelResult KernelRunner::runKernelWithManipulator(const Kernel& kernel, TuningManipulator* manipulator, const KernelConfiguration& configuration,
    const std::vector<OutputDescriptor>& output)
{
    KernelId kernelId = kernel.getId();
    std::string source = kernelManager->getKernelSourceWithDefines(kernelId, configuration);
    KernelRuntimeData kernelData(kernelId, kernel.getName(), source, configuration.getGlobalSize(), configuration.getLocalSize(),
        kernel.getArgumentIds(), configuration.getLocalMemoryModifiers());

    manipulator->manipulatorInterface = manipulatorInterfaceImplementation.get();
    manipulatorInterfaceImplementation->addKernel(kernelId, kernelData);
    manipulatorInterfaceImplementation->setConfiguration(configuration);
    manipulatorInterfaceImplementation->setKernelArguments(argumentManager->getArguments(kernel.getArgumentIds()));
    uint64_t manipulatorDuration;

    if (kernelProfilingFlag)
    {
        manipulatorInterfaceImplementation->setProfiledKernels(std::set<KernelId>{kernelId});
        uint64_t remainingCount = computeEngine->getRemainingKernelProfilingRuns(kernelData.getName(), kernelData.getSource());
        while (remainingCount > 0)
        {
            manipulatorDuration = launchManipulator(kernelId, manipulator);
            uint64_t newCount = computeEngine->getRemainingKernelProfilingRuns(kernelData.getName(), kernelData.getSource());

            if (newCount == remainingCount)
            {
                throw std::runtime_error(
                    std::string("Tuning manipulator does not collect any kernel profiling data for kernel with the following id: ")
                    + std::to_string(kernelId));
            }

            if (newCount != 0)
            {
                manipulatorInterfaceImplementation->resetOverhead();
                computeEngine->clearBuffers();
            }

            remainingCount = newCount;
        }
    }
    else
    {
        manipulatorDuration = launchManipulator(kernelId, manipulator);
    }

    manipulatorInterfaceImplementation->downloadBuffers(output);
    KernelResult result = manipulatorInterfaceImplementation->getCurrentResult();
    manipulatorInterfaceImplementation->clearData();
    manipulator->manipulatorInterface = nullptr;

    manipulatorDuration -= result.getOverhead();
    result.setKernelName(kernel.getName());
    result.setComputationDuration(manipulatorDuration);
    return result;
}

KernelResult KernelRunner::runCompositionWithManipulator(const KernelComposition& composition, TuningManipulator* manipulator,
    const KernelConfiguration& configuration, const std::vector<OutputDescriptor>& output)
{
    const KernelId compositionId = composition.getId();
    manipulator->manipulatorInterface = manipulatorInterfaceImplementation.get();
    std::vector<KernelArgument*> allArguments = argumentManager->getArguments(composition.getSharedArgumentIds());
    std::vector<KernelRuntimeData> compositionData;

    for (const auto* kernel : composition.getKernels())
    {
        KernelId kernelId = kernel->getId();
        std::vector<ArgumentId> argumentIds = composition.getKernelArgumentIds(kernelId);
        std::string source = kernelManager->getKernelSourceWithDefines(kernelId, configuration);

        KernelRuntimeData kernelData(kernelId, kernel->getName(), source, configuration.getCompositionKernelGlobalSize(kernelId),
            configuration.getCompositionKernelLocalSize(kernelId), argumentIds, configuration.getCompositionKernelLocalMemoryModifiers(kernelId));
        manipulatorInterfaceImplementation->addKernel(kernelId, kernelData);
        compositionData.push_back(kernelData);

        std::vector<KernelArgument*> newArguments = argumentManager->getArguments(argumentIds);
        for (const auto newArgument : newArguments)
        {
            if (!elementExists(newArgument, allArguments))
            {
                allArguments.push_back(newArgument);
            }
        }
    }

    manipulatorInterfaceImplementation->setConfiguration(configuration);
    manipulatorInterfaceImplementation->setKernelArguments(allArguments);
    uint64_t manipulatorDuration;

    if (kernelProfilingFlag)
    {
        manipulatorInterfaceImplementation->setProfiledKernels(composition.getProfiledKernels());
        uint64_t remainingCount = getRemainingKernelProfilingRunsForComposition(composition, compositionData);
        while (remainingCount > 0)
        {
            manipulatorDuration = launchManipulator(compositionId, manipulator);
            uint64_t newCount = getRemainingKernelProfilingRunsForComposition(composition, compositionData);

            if (newCount == remainingCount)
            {
                throw std::runtime_error(
                    std::string("Tuning manipulator does not collect any kernel profiling data for composition with the following id: ")
                    + std::to_string(compositionId));
            }

            if (newCount != 0)
            {
                manipulatorInterfaceImplementation->resetOverhead();
                computeEngine->clearBuffers();
            }

            remainingCount = newCount;
        }
    }
    else
    {
        manipulatorDuration = launchManipulator(compositionId, manipulator);
    }

    manipulatorInterfaceImplementation->downloadBuffers(output);
    KernelResult result = manipulatorInterfaceImplementation->getCurrentResult();
    manipulatorInterfaceImplementation->clearData();
    manipulator->manipulatorInterface = nullptr;

    manipulatorDuration -= result.getOverhead();
    result.setKernelName(composition.getName());
    result.setComputationDuration(manipulatorDuration);
    return result;
}

uint64_t KernelRunner::launchManipulator(const KernelId kernelId, TuningManipulator* manipulator)
{
    if (manipulator->enableArgumentPreload())
    {
        manipulatorInterfaceImplementation->uploadBuffers();
    }

    Timer timer;
    try
    {
        timer.start();
        manipulator->launchComputation(kernelId);
        if (disabledSynchronizationManipulators.find(kernelId) == disabledSynchronizationManipulators.end())
        {
            manipulatorInterfaceImplementation->synchronizeDeviceInternal();
        }
        timer.stop();
    }
    catch (const std::runtime_error&)
    {
        manipulatorInterfaceImplementation->synchronizeDeviceInternal();
        manipulatorInterfaceImplementation->clearData();
        manipulator->manipulatorInterface = nullptr;
        throw;
    }

    return timer.getElapsedTime();
}

uint64_t KernelRunner::getRemainingKernelProfilingRunsForComposition(const KernelComposition& composition,
    const std::vector<KernelRuntimeData>& compositionData)
{
    uint64_t count = 0;
    const std::set<KernelId>& profilingKernels = composition.getProfiledKernels();

    for (const auto& kernelData : compositionData)
    {
        if (profilingKernels.find(kernelData.getId()) != profilingKernels.end())
        {
            count += computeEngine->getRemainingKernelProfilingRuns(kernelData.getName(), kernelData.getSource());
        }
    }

    return count;
}

void KernelRunner::validateResult(const Kernel& kernel, KernelResult& result, const KernelRunMode mode)
{
    if (!result.isValid())
    {
        return;
    }

    const bool resultIsCorrect = resultValidator.validateArguments(kernel, mode);

    if (resultIsCorrect)
    {
        Logger::logInfo(std::string("Kernel run completed successfully in ") + std::to_string((result.getComputationDuration()) / 1'000'000) + "ms");
        result.setValid(true);
    }
    else
    {
        Logger::logWarning(std::string("Kernel run completed in ") + std::to_string((result.getComputationDuration()) / 1'000'000)
            + "ms, but results differ");
        result.setErrorMessage("Results differ");
        result.setValid(false);
    }
}

} // namespace ktt
