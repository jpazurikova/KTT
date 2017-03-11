#pragma once

#include <string>

#include "CL/cl.h"
#include "../enums/kernel_argument_access_type.h"

namespace ktt
{

std::string getOpenCLEnumName(const cl_int value);
void checkOpenCLError(const cl_int value);
void checkOpenCLError(const cl_int value, const std::string& message);
cl_mem_flags getOpenCLMemoryType(const KernelArgumentAccessType& kernelArgumentAccessType);
cl_ulong getKernelExecutionDuration(const cl_event profilingEvent);

} // namespace ktt
