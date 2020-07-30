-- Configuration variables
ktt_library_name = "ktt"
cuda_projects = false
opencl_projects = false
vulkan_projects = false

-- Helper functions to find compute API headers and libraries
function findLibrariesAmd()
    local path = os.getenv("AMDAPPSDKROOT")
    
    if not path then
        return false
    end
    
    defines { "KTT_PLATFORM_AMD" }
    includedirs { "$(AMDAPPSDKROOT)/include" }
        
    if os.target() == "linux" then
        libdirs { "$(AMDAPPSDKROOT)/lib64" }
    else
        libdirs { "$(AMDAPPSDKROOT)/lib/x86_64" }
    end
    
    if not _OPTIONS["no-opencl"] then
        opencl_projects = true
        defines { "KTT_PLATFORM_OPENCL" }
        links { "OpenCL" }
        
        if _OPTIONS["profiling"] == "gpa" then
            defines { "KTT_PROFILING_GPA" }
            includedirs { "libraries/include" }
            
            if os.target() == "linux" then
                libdirs { "libraries/lib/linux/gpu_perf_api" }
            else
                -- One of the GPA headers includes Windows.h with evil min/max macros
                defines { "NOMINMAX" }
                libdirs { "libraries/lib/windows/gpu_perf_api" }
            end  
        end
        
        if _OPTIONS["profiling"] == "gpa-legacy" then
            defines { "KTT_PROFILING_GPA_LEGACY" }
            includedirs { "libraries/include" }
            
            if os.target() == "linux" then
                libdirs { "libraries/lib/linux/gpu_perf_api_legacy" }
            else
                defines { "NOMINMAX" }
                libdirs { "libraries/lib/windows/gpu_perf_api_legacy" }
            end  
        end
    end
    
    return true
end

function findLibrariesIntel()
    local path = os.getenv("INTELOCLSDKROOT")
    
    if not path then
        return false
    end
    
    defines { "KTT_PLATFORM_INTEL" }
    includedirs { "$(INTELOCLSDKROOT)/include" }
        
    if os.target() == "linux" then
        libdirs { "$(INTELOCLSDKROOT)/lib64" }
    else
        libdirs { "$(INTELOCLSDKROOT)/lib/x64" }
    end
    
    if not _OPTIONS["no-opencl"] then
        opencl_projects = true
        defines { "KTT_PLATFORM_OPENCL" }
        links { "OpenCL" }
    end
    
    return true
end

function findLibrariesNvidia()
    local path = os.getenv("CUDA_PATH")
    
    if not path then
        return false
    end
    
    defines { "KTT_PLATFORM_NVIDIA" }
    includedirs { "$(CUDA_PATH)/include" }
        
    if os.target() == "linux" then
        libdirs { "$(CUDA_PATH)/lib64" }
    else
        libdirs { "$(CUDA_PATH)/lib/x64" }
    end
    
    if not _OPTIONS["no-opencl"] then
        opencl_projects = true
        defines { "KTT_PLATFORM_OPENCL" }
        links { "OpenCL" }
    end
        
    if not _OPTIONS["no-cuda"] then
        cuda_projects = true
        defines { "KTT_PLATFORM_CUDA" }
        links { "cuda", "nvrtc" }
        
        if _OPTIONS["profiling"] == "cupti-legacy" or _OPTIONS["profiling"] == "cupti" then
            includedirs { "$(CUDA_PATH)/extras/CUPTI/include" }
            libdirs { "$(CUDA_PATH)/extras/CUPTI/lib64" }
            links { "cupti" }
        end
        
        if _OPTIONS["profiling"] == "cupti-legacy" then
            defines { "KTT_PROFILING_CUPTI_LEGACY" }
            
            if os.target() == "windows" then
                libdirs { "$(CUDA_PATH)/extras/CUPTI/libx64" }
            end
        end
        
        if _OPTIONS["profiling"] == "cupti" then
            defines { "KTT_PROFILING_CUPTI" }
            links { "nvperf_host", "nvperf_target" }
        end
    end
        
    return true
end

function findLibraries()
    if findLibrariesAmd() then
        return true
    end
    
    if findLibrariesIntel() then
        return true
    end
    
    if findLibrariesNvidia() then
        return true
    end
    
    return false
end

function findVulkan()
    local path = os.getenv("VULKAN_SDK")
    
    if not path then
        return false
    end
    
    if os.target() == "linux" then
        includedirs { "$(VULKAN_SDK)/include", "libraries/include" }
        libdirs { "$(VULKAN_SDK)/lib", "libraries/lib/linux/shaderc_ktt" }
    else
        includedirs { "$(VULKAN_SDK)/Include", "libraries/include" }
        libdirs { "$(VULKAN_SDK)/Lib", "libraries/lib/windows/shaderc_ktt" }
    end
    
    vulkan_projects = true
    defines { "KTT_PLATFORM_VULKAN" }
    
    if os.target() == "linux" then
        links { "vulkan", "shaderc_shared" }
    else
        links { "vulkan-1", "shaderc_shared" }
    end
    
    return true
end

-- Command line arguments definition
newoption
{
    trigger = "platform",
    value = "vendor",
    description = "Specifies platform for KTT library compilation",
    allowed =
    {
        { "amd", "AMD" },
        { "intel", "Intel" },
        { "nvidia", "Nvidia" }
    }
}

newoption
{
    trigger = "vulkan",
    description = "Enables compilation of Vulkan backend"
}

newoption
{
    trigger = "profiling",
    value = "library",
    description = "Enables compilation of kernel profiling functionality using specified library",
    allowed =
    {
        { "cupti", "Nvidia CUPTI for Volta and Turing" },
        { "cupti-legacy", "Nvidia CUPTI for legacy GPUs (Volta and older)" },
        { "gpa", "AMD GPA for GCN 3.0 GPUs and newer" },
        { "gpa-legacy", "AMD GPA for GCN 5.0 GPUs and older" }
    }
}

newoption
{
    trigger = "outdir",
    value = "path",
    description = "Specifies output directory for generated files"
}

newoption
{
    trigger = "no-cuda",
    description = "Disables compilation of CUDA backend (Nvidia platform only)"
}

newoption
{
    trigger = "no-opencl",
    description = "Disables compilation of OpenCL backend"
}

newoption
{
    trigger = "tests",
    description = "Enables compilation of unit tests"
}

newoption
{
    trigger = "no-examples",
    description = "Disables compilation of examples"
}

newoption
{
    trigger = "no-tutorials",
    description = "Disables compilation of tutorials"
}

-- Project configuration
workspace "ktt"
    local buildPath = "build"
    if _OPTIONS["outdir"] then
        buildPath = _OPTIONS["outdir"]
    end
    
    configurations { "Release", "Debug" }
    platforms { "x86_64" }
    architecture "x86_64"
    
    location(buildPath)
    language "C++"
    cppdialect "C++14"
    
    filter "configurations:Debug"
        defines { "KTT_CONFIGURATION_DEBUG" }
        symbols "On"
    
    filter "configurations:Release"
        defines { "KTT_CONFIGURATION_RELEASE" }
        optimize "On"
    
    filter {}
    
    targetdir(buildPath .. "/%{cfg.platform}_%{cfg.buildcfg}")
    objdir(buildPath .. "/%{cfg.platform}_%{cfg.buildcfg}/obj")
    
-- Library configuration
project "ktt"
    kind "SharedLib"
    files { "source/**.h", "source/**.hpp", "source/**.cpp" }
    includedirs { "source" }
    defines { "KTT_LIBRARY" }
    targetname(ktt_library_name)

    local libraries = false
    
    if _OPTIONS["platform"] then
        if _OPTIONS["platform"] == "amd" then
            libraries = findLibrariesAmd()
        elseif _OPTIONS["platform"] == "intel" then
            libraries = findLibrariesIntel()
        else
            libraries = findLibrariesNvidia()
        end
    else
        libraries = findLibraries()
    end
    
    if not libraries and (not _OPTIONS["vulkan"] or _OPTIONS["platform"]) then
        error("Compute API libraries were not found. Please ensure that path to your device vendor SDK is correctly set in the environment variables:\nAMDAPPSDKROOT for AMD\nINTELOCLSDKROOT for Intel\nCUDA_PATH for Nvidia")
    end
    
    if _OPTIONS["vulkan"] then
        vulkan = findVulkan()
        
        if not vulkan then
            error("Vulkan SDK was not found. Please ensure that path to the SDK is correctly set in the environment variables under VULKAN_SDK.")
        end
    end
    
-- Examples configuration 
if not _OPTIONS["no-examples"] then

if opencl_projects then

project "bicg_opencl"
    kind "ConsoleApp"
    files { "examples/bicg/*.cpp", "examples/bicg/*.cl" }
    includedirs { "source" }
    defines { "KTT_OPENCL_EXAMPLE" }
    links { "ktt" }

project "conv_opencl"
    kind "ConsoleApp"
    files { "examples/cltune-conv/*.cpp", "examples/cltune-conv/*.cl" }
    includedirs { "source" }
    defines { "KTT_OPENCL_EXAMPLE" }
    links { "ktt" }

project "gemm_opencl"
    kind "ConsoleApp"
    files { "examples/cltune-gemm/*.cpp", "examples/cltune-gemm/*.cl" }
    includedirs { "source" }
    defines { "KTT_OPENCL_EXAMPLE" }
    links { "ktt" }

project "conv_3d_opencl"
    kind "ConsoleApp"
    files { "examples/conv_3d/*.cpp", "examples/conv_3d/*.cl" }
    includedirs { "source" }
    defines { "KTT_OPENCL_EXAMPLE" }
    links { "ktt" }

project "coulomb_sum_2d_opencl"
    kind "ConsoleApp"
    files { "examples/coulomb_sum_2d/*.cpp", "examples/coulomb_sum_2d/*.cl" }
    includedirs { "source" }
    defines { "KTT_OPENCL_EXAMPLE" }
    links { "ktt" }

project "coulomb_sum_3d_opencl"
    kind "ConsoleApp"
    files { "examples/coulomb_sum_3d/*.cpp", "examples/coulomb_sum_3d/*.cl" }
    includedirs { "source" }
    defines { "KTT_OPENCL_EXAMPLE" }
    links { "ktt" }

project "coulomb_sum_3d_iterative_opencl"
    kind "ConsoleApp"
    files { "examples/coulomb_sum_3d_iterative/*.cpp", "examples/coulomb_sum_3d_iterative/*.cl" }
    includedirs { "source" }
    defines { "KTT_OPENCL_EXAMPLE" }
    links { "ktt" }

project "covariance_opencl"
    kind "ConsoleApp"
    files { "examples/covariance/*.cpp", "examples/covariance/*.cl" }
    includedirs { "source" }
    defines { "KTT_OPENCL_EXAMPLE" }
    links { "ktt" }

project "gemm_demo_opencl"
    kind "ConsoleApp"
    files { "examples/gemm_batch/demo.cpp", "examples/gemm_batch/*.cl" }
    includedirs { "source" }
    defines { "KTT_OPENCL_EXAMPLE" }
    links { "ktt" }

project "gemm_batch_opencl"
    kind "ConsoleApp"
    files { "examples/gemm_batch/gemm_batch.cpp", "examples/gemm_batch/*.cl" }
    includedirs { "source" }
    defines { "KTT_OPENCL_EXAMPLE" }
    links { "ktt" }

project "mtran_opencl"
    kind "ConsoleApp"
    files { "examples/mtran/*.cpp", "examples/mtran/*.cl" }
    includedirs { "source" }
    defines { "KTT_OPENCL_EXAMPLE" }
    links { "ktt" }

project "nbody_opencl"
    kind "ConsoleApp"
    files { "examples/nbody/*.cpp", "examples/nbody/*.cl" }
    includedirs { "source" }
    defines { "KTT_OPENCL_EXAMPLE" }
    links { "ktt" }

project "reduction_opencl"
    kind "ConsoleApp"
    files { "examples/reduction/*.h", "examples/reduction/*.cpp", "examples/reduction/*.cl" }
    includedirs { "source" }
    defines { "KTT_OPENCL_EXAMPLE" }
    links { "ktt" }

if os.target() == "linux" then
project "hotspot_opencl"
    kind "ConsoleApp"
    files { "examples/rodinia-hotspot/*.h", "examples/rodinia-hotspot/*.cpp", "examples/rodinia-hotspot/*.cl" }
    includedirs { "source" }
    defines { "KTT_OPENCL_EXAMPLE" }
    links { "ktt" }
end

project "sort_opencl"
    kind "ConsoleApp"
    files { "examples/sort/*.h", "examples/sort/*.cpp", "examples/sort/*.cl" }
    includedirs { "source" }
    defines { "KTT_OPENCL_EXAMPLE" }
    links { "ktt" }

project "sort_new_opencl"
    kind "ConsoleApp"
    files {"examples/sort-new/*.h", "examples/sort-new/*.cpp", "examples/sort-new/*.cl"}
    includedirs { "source" }
    defines { "KTT_OPENCL_EXAMPLE" }
    links { "ktt" }

project "memory_map_test_opencl"
    kind "ConsoleApp"
    files {"examples/memory-map-test/*.h", "examples/memory-map-test/*.cpp", "examples/memory-map-test/*.cl"}
    includedirs { "source" }
    defines { "KTT_OPENCL_EXAMPLE" }
    links { "ktt" }

end -- opencl_projects

if cuda_projects then

project "bicg_cuda"
    kind "ConsoleApp"
    files { "examples/bicg/*.cpp", "examples/bicg/*.cu" }
    includedirs { "source" }
    defines { "KTT_CUDA_EXAMPLE" }
    links { "ktt" }
    
project "conv_cuda"
    kind "ConsoleApp"
    files { "examples/cltune-conv/*.cpp", "examples/cltune-conv/*.cu" }
    includedirs { "source" }
    defines { "KTT_CUDA_EXAMPLE" }
    links { "ktt" }
    
project "gemm_cuda"
    kind "ConsoleApp"
    files { "examples/cltune-gemm/*.cpp", "examples/cltune-gemm/*.cu" }
    includedirs { "source" }
    defines { "KTT_CUDA_EXAMPLE" }
    links { "ktt" }
    
project "coulomb_sum_3d_cuda"
    kind "ConsoleApp"
    files { "examples/coulomb_sum_3d/*.cpp", "examples/coulomb_sum_3d/*.cu" }
    includedirs { "source" }
    defines { "KTT_CUDA_EXAMPLE" }
    links { "ktt" }
    
project "gemm_demo_cuda"
    kind "ConsoleApp"
    files { "examples/gemm_batch/demo.cpp", "examples/gemm_batch/*.cu" }
    includedirs { "source" }
    defines { "KTT_CUDA_EXAMPLE" }
    links { "ktt" }
    
project "gemm_batch_cuda"
    kind "ConsoleApp"
    files { "examples/gemm_batch/gemm_batch.cpp", "examples/gemm_batch/*.cu" }
    includedirs { "source" }
    defines { "KTT_CUDA_EXAMPLE" }
    links { "ktt" }

project "mtran_cuda"
    kind "ConsoleApp"
    files { "examples/mtran/*.cpp", "examples/mtran/*.cu" }
    includedirs { "source" }
    defines { "KTT_CUDA_EXAMPLE" }
    links { "ktt" }

project "nbody_cuda"
    kind "ConsoleApp"
    files { "examples/nbody/*.cpp", "examples/nbody/*.cu" }
    includedirs { "source" }
    defines { "KTT_CUDA_EXAMPLE" }
    links { "ktt" }

project "reduction_cuda"
    kind "ConsoleApp"
    files { "examples/reduction/*.h", "examples/reduction/*.cpp", "examples/reduction/*.cu" }
    includedirs { "source" }
    defines { "KTT_CUDA_EXAMPLE" }
    links { "ktt" }

if os.target() == "linux" then
project "hotspot_cuda"
    kind "ConsoleApp"
    files { "examples/rodinia-hotspot/*.h", "examples/rodinia-hotspot/*.cpp", "examples/rodinia-hotspot/*.cu" }
    includedirs { "source" }
    defines { "KTT_CUDA_EXAMPLE" }
    links { "ktt" }
end

project "sort_cuda"
    kind "ConsoleApp"
    files { "examples/sort/*.h", "examples/sort/*.cpp", "examples/sort/*.cu" }
    includedirs { "source" }
    defines { "KTT_CUDA_EXAMPLE" }
    links { "ktt" }

project "sort_new_cuda"
    kind "ConsoleApp"
    files {"examples/sort-new/*.h", "examples/sort-new/*.cpp", "examples/sort-new/*.cu"}
    includedirs { "source" }
    defines { "KTT_CUDA_EXAMPLE" }
    links { "ktt" }

project "memory_map_test_cuda"
    kind "ConsoleApp"
    files {"examples/memory-map-test/*.h", "examples/memory-map-test/*.cpp", "examples/memory-map-test/*.cu"}
    includedirs { "source" }
    defines { "KTT_CUDA_EXAMPLE" }
    links { "ktt" }
end -- cuda_projects

end -- _OPTIONS["no-examples"]

-- Tutorials configuration 
if not _OPTIONS["no-tutorials"] then

if opencl_projects then
project "00_info_opencl"
    kind "ConsoleApp"
    files { "tutorials/00_compute_api_info/compute_api_info_opencl.cpp" }
    includedirs { "source" }
    links { "ktt" }

project "01_running_kernel_opencl"
    kind "ConsoleApp"
    files { "tutorials/01_running_kernel/running_kernel_opencl.cpp", "tutorials/01_running_kernel/opencl_kernel.cl" }
    includedirs { "source" }
    links { "ktt" }

project "02_tuning_kernel_simple_opencl"
    kind "ConsoleApp"
    files { "tutorials/02_tuning_kernel_simple/tuning_kernel_simple_opencl.cpp", "tutorials/02_tuning_kernel_simple/opencl_kernel.cl" }
    includedirs { "source" }
    links { "ktt" }
    
project "03_custom_kernel_arguments_opencl"
    kind "ConsoleApp"
    files { "tutorials/03_custom_kernel_arguments/custom_kernel_arguments_opencl.cpp", "tutorials/03_custom_kernel_arguments/opencl_kernel.cl" }
    includedirs { "source" }
    links { "ktt" }
end -- opencl_projects
    
if cuda_projects then
project "00_info_cuda"
    kind "ConsoleApp"
    files { "tutorials/00_compute_api_info/compute_api_info_cuda.cpp" }
    includedirs { "source" }
    links { "ktt" }

project "01_running_kernel_cuda"
    kind "ConsoleApp"
    files { "tutorials/01_running_kernel/running_kernel_cuda.cpp", "tutorials/01_running_kernel/cuda_kernel.cu" }
    includedirs { "source" }
    links { "ktt" }

project "02_tuning_kernel_simple_cuda"
    kind "ConsoleApp"
    files { "tutorials/02_tuning_kernel_simple/tuning_kernel_simple_cuda.cpp", "tutorials/02_tuning_kernel_simple/cuda_kernel.cu" }
    includedirs { "source" }
    links { "ktt" }
   
project "03_custom_kernel_arguments_cuda"
    kind "ConsoleApp"
    files { "tutorials/03_custom_kernel_arguments/custom_kernel_arguments_cuda.cpp", "tutorials/03_custom_kernel_arguments/cuda_kernel.cu" }
    includedirs { "source" }
    links { "ktt" }
end -- cuda_projects

if vulkan_projects then
project "00_info_vulkan"
    kind "ConsoleApp"
    files { "tutorials/00_compute_api_info/compute_api_info_vulkan.cpp" }
    includedirs { "source" }
    links { "ktt" }
    
project "01_running_kernel_vulkan"
    kind "ConsoleApp"
    files { "tutorials/01_running_kernel/running_kernel_vulkan.cpp", "tutorials/01_running_kernel/vulkan_kernel.glsl" }
    includedirs { "source" }
    links { "ktt" }
    
project "02_tuning_kernel_simple_vulkan"
    kind "ConsoleApp"
    files { "tutorials/02_tuning_kernel_simple/tuning_kernel_simple_vulkan.cpp", "tutorials/02_tuning_kernel_simple/vulkan_kernel.glsl" }
    includedirs { "source" }
    links { "ktt" }
    
end -- vulkan_projects

end -- _OPTIONS["no-tutorials"]

-- Unit tests configuration   
if _OPTIONS["tests"] then

project "tests"
    kind "ConsoleApp"
    files { "tests/**.hpp", "tests/**.cpp", "tests/**.cl", "source/**.h", "source/**.hpp", "source/**.cpp" }
    includedirs { "tests", "source" }
    defines { "KTT_TESTS", "DO_NOT_USE_WMAIN" }
    
    if _OPTIONS["no-opencl"] then
        removefiles { "tests/opencl_engine_tests.cpp" }
    end
    
    if _OPTIONS["platform"] then
        if _OPTIONS["platform"] == "amd" then
            findLibrariesAmd()
        elseif _OPTIONS["platform"] == "intel" then
            findLibrariesIntel()
        else
            findLibrariesNvidia()
        end
    else
        findLibraries()
    end
    
    if _OPTIONS["vulkan"] then
        findVulkan()
    end
end -- _OPTIONS["tests"]
