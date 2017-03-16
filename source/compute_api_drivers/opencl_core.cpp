#include "opencl_core.h"

#include "CL/cl.h"

namespace ktt
{

OpenCLCore::OpenCLCore(const size_t platformIndex, const size_t deviceIndex):
    compilerOptions(std::string(""))
{
    auto platforms = getOpenCLPlatforms();
    if (platformIndex >= platforms.size())
    {
        throw std::runtime_error("Invalid platform index: " + platformIndex);
    }

    auto devices = getOpenCLDevices(platforms.at(platformIndex));
    if (deviceIndex >= devices.size())
    {
        throw std::runtime_error("Invalid device index: " + deviceIndex);
    }

    cl_device_id device = devices.at(deviceIndex).getId();
    context = std::make_unique<OpenCLContext>(platforms.at(platformIndex).getId(), std::vector<cl_device_id> { device });
    commandQueue = std::make_unique<OpenCLCommandQueue>(context->getContext(), device);
}

void OpenCLCore::printOpenCLInfo(std::ostream& outputTarget)
{
    auto platforms = getOpenCLPlatforms();

    for (size_t i = 0; i < platforms.size(); i++)
    {
        outputTarget << "Platform " << i << ": " << platforms.at(i).getName() << std::endl;
        auto devices = getOpenCLDevices(platforms.at(i));

        outputTarget << "Devices for platform " << i << ":" << std::endl;
        for (size_t j = 0; j < devices.size(); j++)
        {
            outputTarget << "Device " << j << ": " << devices.at(j).getName() << std::endl;
        }
        outputTarget << std::endl;
    }
}

PlatformInfo OpenCLCore::getOpenCLPlatformInfo(const size_t platformIndex)
{
    auto platforms = getOpenCLPlatforms();
    PlatformInfo result(platformIndex, platforms.at(platformIndex).getName());

    cl_platform_id id = platforms.at(platformIndex).getId();
    result.setExtensions(getPlatformInfo(id, CL_PLATFORM_EXTENSIONS));
    result.setVendor(getPlatformInfo(id, CL_PLATFORM_VENDOR));
    result.setVersion(getPlatformInfo(id, CL_PLATFORM_VERSION));

    return result;
}

std::vector<PlatformInfo> OpenCLCore::getOpenCLPlatformInfoAll()
{
    std::vector<PlatformInfo> result;
    auto platforms = getOpenCLPlatforms();

    for (size_t i = 0; i < platforms.size(); i++)
    {
        result.push_back(getOpenCLPlatformInfo(i));
    }

    return result;
}

DeviceInfo OpenCLCore::getOpenCLDeviceInfo(const size_t platformIndex, const size_t deviceIndex)
{
    auto platforms = getOpenCLPlatforms();
    auto devices = getOpenCLDevices(platforms.at(platformIndex));
    DeviceInfo result(deviceIndex, devices.at(deviceIndex).getName());

    cl_device_id id = devices.at(deviceIndex).getId();
    result.setExtensions(getDeviceInfo(id, CL_DEVICE_EXTENSIONS));
    result.setVendor(getDeviceInfo(id, CL_DEVICE_VENDOR));
        
    uint64_t globalMemorySize;
    checkOpenCLError(clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(uint64_t), &globalMemorySize, nullptr));
    result.setGlobalMemorySize(globalMemorySize);

    uint64_t localMemorySize;
    checkOpenCLError(clGetDeviceInfo(id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(uint64_t), &localMemorySize, nullptr));
    result.setLocalMemorySize(localMemorySize);

    uint64_t maxConstantBufferSize;
    checkOpenCLError(clGetDeviceInfo(id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(uint64_t), &maxConstantBufferSize, nullptr));
    result.setMaxConstantBufferSize(maxConstantBufferSize);

    uint32_t maxComputeUnits;
    checkOpenCLError(clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uint32_t), &maxComputeUnits, nullptr));
    result.setMaxComputeUnits(maxComputeUnits);

    size_t maxWorkGroupSize;
    checkOpenCLError(clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, nullptr));
    result.setMaxWorkGroupSize(maxWorkGroupSize);

    cl_device_type deviceType;
    checkOpenCLError(clGetDeviceInfo(id, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
    result.setDeviceType(getDeviceType(deviceType));

    return result;
}

std::vector<DeviceInfo> OpenCLCore::getOpenCLDeviceInfoAll(const size_t platformIndex)
{
    std::vector<DeviceInfo> result;
    auto platforms = getOpenCLPlatforms();
    auto devices = getOpenCLDevices(platforms.at(platformIndex));

    for (size_t i = 0; i < devices.size(); i++)
    {
        result.push_back(getOpenCLDeviceInfo(platformIndex, i));
    }

    return result;
}

void OpenCLCore::setOpenCLCompilerOptions(const std::string& options)
{
    compilerOptions = options;
}

OpenCLProgram OpenCLCore::createAndBuildProgram(const std::string& source) const
{
    OpenCLProgram program(source, context->getContext(), context->getDevices());
    buildProgram(program);
    return program;
}

OpenCLBuffer OpenCLCore::createBuffer(const ArgumentMemoryType& argumentMemoryType, const size_t size) const
{
    OpenCLBuffer buffer(context->getContext(), getOpenCLMemoryType(argumentMemoryType), size);
    return buffer;
}

void OpenCLCore::updateBuffer(OpenCLBuffer& buffer, const void* source, const size_t dataSize) const
{
    cl_int result = clEnqueueWriteBuffer(commandQueue->getQueue(), buffer.getBuffer(), CL_TRUE, 0, dataSize, source, 0, nullptr, nullptr);
    checkOpenCLError(result);
}

void OpenCLCore::getBufferData(const OpenCLBuffer& buffer, void* destination, const size_t dataSize) const
{
    cl_int result = clEnqueueReadBuffer(commandQueue->getQueue(), buffer.getBuffer(), CL_TRUE, 0, dataSize, destination, 0, nullptr, nullptr);
    checkOpenCLError(result);
}

OpenCLKernel OpenCLCore::createKernel(const OpenCLProgram& program, const std::string& kernelName) const
{
    OpenCLKernel kernel(program.getProgram(), kernelName);
    return kernel;
}

void OpenCLCore::setKernelArgument(OpenCLKernel& kernel, const OpenCLBuffer& buffer) const
{
    kernel.setKernelArgument(buffer.getBuffer(), buffer.getSize());
}

cl_ulong OpenCLCore::runKernel(OpenCLKernel& kernel, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize) const
{
    cl_event profilingEvent;
    cl_int result = clEnqueueNDRangeKernel(commandQueue->getQueue(), kernel.getKernel(), globalSize.size(), nullptr, globalSize.data(),
        localSize.data(), 0, nullptr, &profilingEvent);
    checkOpenCLError(result);

    clFinish(commandQueue->getQueue());
    return getKernelRunDuration(profilingEvent);
}

std::vector<OpenCLPlatform> OpenCLCore::getOpenCLPlatforms()
{
    cl_uint platformCount;
    checkOpenCLError(clGetPlatformIDs(0, nullptr, &platformCount));

    std::vector<cl_platform_id> platformIds(platformCount);
    checkOpenCLError(clGetPlatformIDs(platformCount, platformIds.data(), nullptr));

    std::vector<OpenCLPlatform> platforms;
    for (const auto platformId : platformIds)
    {
        std::string name = getPlatformInfo(platformId, CL_PLATFORM_NAME);
        platforms.push_back(OpenCLPlatform(platformId, name));
    }

    return platforms;
}

std::vector<OpenCLDevice> OpenCLCore::getOpenCLDevices(const OpenCLPlatform& platform)
{
    cl_uint deviceCount;
    checkOpenCLError(clGetDeviceIDs(platform.getId(), CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount));

    std::vector<cl_device_id> deviceIds(deviceCount);
    checkOpenCLError(clGetDeviceIDs(platform.getId(), CL_DEVICE_TYPE_ALL, deviceCount, deviceIds.data(), nullptr));

    std::vector<OpenCLDevice> devices;
    for (const auto deviceId : deviceIds)
    {
        std::string name = getDeviceInfo(deviceId, CL_DEVICE_NAME);
        devices.push_back(OpenCLDevice(deviceId, name));
    }

    return devices;
}

std::string OpenCLCore::getPlatformInfo(const cl_platform_id id, const cl_platform_info info)
{
    size_t infoSize;
    checkOpenCLError(clGetPlatformInfo(id, info, 0, nullptr, &infoSize));
    std::string infoString(infoSize, ' ');
    checkOpenCLError(clGetPlatformInfo(id, info, infoSize, &infoString[0], nullptr));
    
    return infoString;
}

std::string OpenCLCore::getDeviceInfo(const cl_device_id id, const cl_device_info info)
{
    size_t infoSize;
    checkOpenCLError(clGetDeviceInfo(id, info, 0, nullptr, &infoSize));
    std::string infoString(infoSize, ' ');
    checkOpenCLError(clGetDeviceInfo(id, info, infoSize, &infoString[0], nullptr));

    return infoString;
}

DeviceType OpenCLCore::getDeviceType(const cl_device_type deviceType)
{
    switch (deviceType)
    {
    case CL_DEVICE_TYPE_CPU:
        return DeviceType::CPU;
    case CL_DEVICE_TYPE_GPU:
        return DeviceType::GPU;
    case CL_DEVICE_TYPE_ACCELERATOR:
        return DeviceType::ACCELERATOR;
    case CL_DEVICE_TYPE_DEFAULT:
        return DeviceType::DEFAULT;
    default:
        return DeviceType::CUSTOM;
    }
}

void OpenCLCore::buildProgram(OpenCLProgram& program) const
{
    cl_int result = clBuildProgram(program.getProgram(), program.getDevices().size(), program.getDevices().data(), &compilerOptions[0], nullptr,
        nullptr);
    std::string buildInfo = getProgramBuildInfo(program.getProgram(), program.getDevices().at(0));
    checkOpenCLError(result, buildInfo);
}

std::string OpenCLCore::getProgramBuildInfo(const cl_program program, const cl_device_id id) const
{
    size_t infoSize;
    checkOpenCLError(clGetProgramBuildInfo(program, id, 0, 0, nullptr, &infoSize));
    std::string infoString(infoSize, ' ');
    checkOpenCLError(clGetProgramBuildInfo(program, id, CL_PROGRAM_BUILD_LOG, infoSize, &infoString[0], nullptr));

    return infoString;
}

} // namespace ktt
