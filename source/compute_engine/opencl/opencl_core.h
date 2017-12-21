#pragma once

#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <vector>
#include "opencl_buffer.h"
#include "opencl_command_queue.h"
#include "opencl_context.h"
#include "opencl_device.h"
#include "opencl_event.h"
#include "opencl_kernel.h"
#include "opencl_platform.h"
#include "opencl_program.h"
#include "compute_engine/compute_engine.h"

namespace ktt
{

class OpenclCore : public ComputeEngine
{
public:
    // Constructor
    explicit OpenclCore(const size_t platformIndex, const size_t deviceIndex, const size_t queueCount);

    // Kernel execution method
    KernelResult runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
        const std::vector<ArgumentOutputDescriptor>& outputDescriptors) override;
    void runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers, const QueueId queue,
        const bool synchronizeFlag) override;

    // Utility methods
    void setCompilerOptions(const std::string& options) override;
    void setGlobalSizeType(const GlobalSizeType& type) override;
    void setAutomaticGlobalSizeCorrection(const bool flag) override;
    
    // Queue handling methods
    QueueId getDefaultQueue() const override;
    std::vector<QueueId> getAllQueues() const override;
    void synchronizeQueue(const QueueId queue) override;
    void synchronizeDevice() override;

    // Argument handling methods
    void uploadArgument(KernelArgument& kernelArgument) override;
    void uploadArgument(KernelArgument& kernelArgument, const QueueId queue, const bool synchronizeFlag) override;
    void updateArgument(const ArgumentId id, const void* data, const size_t dataSizeInBytes) override;
    void updateArgument(const ArgumentId id, const void* data, const size_t dataSizeInBytes, const QueueId queue,
        const bool synchronizeFlag) override;
    void downloadArgument(const ArgumentId id, void* destination) const override;
    void downloadArgument(const ArgumentId id, void* destination, const QueueId queue, const bool synchronizeFlag) const override;
    void downloadArgument(const ArgumentId id, void* destination, const size_t dataSizeInBytes) const override;
    void downloadArgument(const ArgumentId id, void* destination, const size_t dataSizeInBytes, const QueueId queue,
        const bool synchronizeFlag) const override;
    KernelArgument downloadArgument(const ArgumentId id) const override;
    void clearBuffer(const ArgumentId id) override;
    void clearBuffers() override;
    void clearBuffers(const ArgumentAccessType& accessType) override;

    // Information retrieval methods
    void printComputeApiInfo(std::ostream& outputTarget) const override;
    std::vector<PlatformInfo> getPlatformInfo() const override;
    std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const override;
    DeviceInfo getCurrentDeviceInfo() const override;

    // Low-level kernel execution methods
    std::unique_ptr<OpenclProgram> createAndBuildProgram(const std::string& source) const;
    void setKernelArgument(OpenclKernel& kernel, KernelArgument& argument);
    cl_ulong enqueueKernel(OpenclKernel& kernel, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize, const QueueId queue,
        const bool synchronizeFlag) const;

private:
    // Attributes
    size_t platformIndex;
    size_t deviceIndex;
    size_t queueCount;
    std::string compilerOptions;
    GlobalSizeType globalSizeType;
    bool globalSizeCorrection;
    std::unique_ptr<OpenclContext> context;
    std::vector<std::unique_ptr<OpenclCommandQueue>> commandQueues;
    std::set<std::unique_ptr<OpenclBuffer>> buffers;

    // Helper methods
    static PlatformInfo getOpenclPlatformInfo(const size_t platformIndex);
    static DeviceInfo getOpenclDeviceInfo(const size_t platformIndex, const size_t deviceIndex);
    static std::vector<OpenclPlatform> getOpenclPlatforms();
    static std::vector<OpenclDevice> getOpenclDevices(const OpenclPlatform& platform);
    static DeviceType getDeviceType(const cl_device_type deviceType);
    OpenclBuffer* findBuffer(const ArgumentId id) const;
    void setKernelArgumentVector(OpenclKernel& kernel, const OpenclBuffer& buffer) const;
    bool loadBufferFromCache(const ArgumentId id, OpenclKernel& openclKernel) const;
};

} // namespace ktt
