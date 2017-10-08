#include <iostream>

#include "tuner_api.h"
#include "tuner_core.h"

namespace ktt
{

Tuner::Tuner(const size_t platformIndex, const size_t deviceIndex) :
    tunerCore(std::make_unique<TunerCore>(platformIndex, deviceIndex, ComputeApi::Opencl, RunMode::Tuning))
{}

Tuner::Tuner(const size_t platformIndex, const size_t deviceIndex, const ComputeApi& computeApi) :
    tunerCore(std::make_unique<TunerCore>(platformIndex, deviceIndex, computeApi, RunMode::Tuning))
{}

Tuner::Tuner(const size_t platformIndex, const size_t deviceIndex, const ComputeApi& computeApi, const RunMode& runMode) :
    tunerCore(std::make_unique<TunerCore>(platformIndex, deviceIndex, computeApi, runMode))
{}

Tuner::~Tuner() = default;

size_t Tuner::addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    return tunerCore->addKernel(source, kernelName, globalSize, localSize);
}

size_t Tuner::addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    try
    {
        return tunerCore->addKernelFromFile(filePath, kernelName, globalSize, localSize);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::setKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIndices)
{
    try
    {
        tunerCore->setKernelArguments(kernelId, argumentIndices);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::addParameter(const size_t kernelId, const std::string& parameterName, const std::vector<size_t>& parameterValues)
{
    try
    {
        tunerCore->addParameter(kernelId, parameterName, parameterValues, ThreadModifierType::None, ThreadModifierAction::Multiply, Dimension::X);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::addParameter(const size_t kernelId, const std::string& parameterName, const std::vector<size_t>& parameterValues,
    const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension)
{
    try
    {
        tunerCore->addParameter(kernelId, parameterName, parameterValues, threadModifierType, threadModifierAction, modifierDimension);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::addConstraint(const size_t kernelId, const std::function<bool(std::vector<size_t>)>& constraintFunction,
    const std::vector<std::string>& parameterNames)
{
    try
    {
        tunerCore->addConstraint(kernelId, constraintFunction, parameterNames);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::setSearchMethod(const size_t kernelId, const SearchMethod& searchMethod, const std::vector<double>& searchArguments)
{
    try
    {
        tunerCore->setSearchMethod(kernelId, searchMethod, searchArguments);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::setTuningManipulator(const size_t kernelId, std::unique_ptr<TuningManipulator> tuningManipulator)
{
    try
    {
        tunerCore->setTuningManipulator(kernelId, std::move(tuningManipulator));
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::enableArgumentPrinting(const size_t argumentId, const std::string& filePath, const ArgumentPrintCondition& argumentPrintCondition)
{
    try
    {
        tunerCore->enableArgumentPrinting(argumentId, filePath, argumentPrintCondition);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
    }
}

void Tuner::tuneKernel(const size_t kernelId)
{
    try
    {
        tunerCore->tuneKernel(kernelId);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::runKernel(const size_t kernelId, const std::vector<ParameterValue>& kernelConfiguration,
    const std::vector<ArgumentOutputDescriptor>& outputDescriptors)
{
    try
    {
        tunerCore->runKernel(kernelId, kernelConfiguration, outputDescriptors);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::setPrintingTimeUnit(const TimeUnit& timeUnit)
{
    tunerCore->setPrintingTimeUnit(timeUnit);
}

void Tuner::setInvalidResultPrinting(const bool flag)
{
    tunerCore->setInvalidResultPrinting(flag);
}

void Tuner::printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const
{
    try
    {
        tunerCore->printResult(kernelId, outputTarget, printFormat);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
    }
}

void Tuner::printResult(const size_t kernelId, const std::string& filePath, const PrintFormat& printFormat) const
{
    try
    {
        tunerCore->printResult(kernelId, filePath, printFormat);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
    }
}

std::vector<ParameterValue> Tuner::getBestConfiguration(const size_t kernelId) const
{
    try
    {
        return tunerCore->getBestConfiguration(kernelId);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::setReferenceKernel(const size_t kernelId, const size_t referenceKernelId,
    const std::vector<ParameterValue>& referenceKernelConfiguration, const std::vector<size_t>& resultArgumentIds)
{
    try
    {
        tunerCore->setReferenceKernel(kernelId, referenceKernelId, referenceKernelConfiguration, resultArgumentIds);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
    }
}

void Tuner::setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<size_t>& resultArgumentIds)
{
    try
    {
        tunerCore->setReferenceClass(kernelId, std::move(referenceClass), resultArgumentIds);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
    }
}

void Tuner::setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold)
{
    try
    {
        tunerCore->setValidationMethod(validationMethod, toleranceThreshold);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
    }
}

void Tuner::setValidationRange(const size_t argumentId, const size_t validationRange)
{
    try
    {
        tunerCore->setValidationRange(argumentId, validationRange);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
    }
}

void Tuner::setCompilerOptions(const std::string& options)
{
    tunerCore->setCompilerOptions(options);
}

void Tuner::printComputeApiInfo(std::ostream& outputTarget) const
{
    try
    {
        tunerCore->printComputeApiInfo(outputTarget);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
    }
}

std::vector<PlatformInfo> Tuner::getPlatformInfo() const
{
    try
    {
        return tunerCore->getPlatformInfo();
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

std::vector<DeviceInfo> Tuner::getDeviceInfo(const size_t platformIndex) const
{
    try
    {
        return tunerCore->getDeviceInfo(platformIndex);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

DeviceInfo Tuner::getCurrentDeviceInfo() const
{
    try
    {
        return tunerCore->getCurrentDeviceInfo();
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

void Tuner::setGlobalSizeType(const GlobalSizeType& globalSizeType)
{
    tunerCore->setGlobalSizeType(globalSizeType);
}

void Tuner::setLoggingTarget(std::ostream& outputTarget)
{
    tunerCore->setLoggingTarget(outputTarget);
}

void Tuner::setLoggingTarget(const std::string& filePath)
{
    tunerCore->setLoggingTarget(filePath);
}

size_t Tuner::addArgument(const void* vectorData, const size_t numberOfElements, const ArgumentDataType& dataType,
    const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType)
{
    try
    {
        return tunerCore->addArgument(vectorData, numberOfElements, dataType, memoryLocation, accessType, ArgumentUploadType::Vector);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

size_t Tuner::addArgument(const void* scalarData, const ArgumentDataType& dataType)
{
    try
    {
        return tunerCore->addArgument(scalarData, 1, dataType, ArgumentMemoryLocation::Device, ArgumentAccessType::ReadOnly,
            ArgumentUploadType::Scalar);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

size_t Tuner::addArgument(const size_t localMemoryElementsCount, const ArgumentDataType& dataType)
{
    try
    {
        return tunerCore->addArgument(nullptr, localMemoryElementsCount, dataType, ArgumentMemoryLocation::Device, ArgumentAccessType::ReadOnly,
            ArgumentUploadType::Local);
    }
    catch (const std::runtime_error& error)
    {
        tunerCore->log(error.what());
        throw;
    }
}

} // namespace ktt
