#pragma once

#include <array>
#include <map>
#include <string>
#include <vector>
#include "kernel.h"

namespace ktt
{

class KernelComposition
{
public:
    // Constructor
    explicit KernelComposition(const KernelId id, const std::string& name, const std::vector<const Kernel*>& kernels);

    // Core methods
    void addParameter(const KernelParameter& parameter);
    void addConstraint(const KernelConstraint& constraint);
    void addParameterPack(const KernelParameterPack& pack);
    void setSharedArguments(const std::vector<ArgumentId>& argumentIds);
    void setThreadModifier(const KernelId id, const ModifierType modifierType, const ModifierDimension modifierDimension,
        const std::vector<std::string>& parameterNames, const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction);
    void setLocalMemoryModifier(const KernelId id, const ArgumentId argumentId, const std::vector<std::string>& parameterNames,
        const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction);
    void setArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds);
    Kernel transformToKernel() const;
    std::map<KernelId, DimensionVector> getModifiedGlobalSizes(const std::vector<ParameterPair>& parameterPairs) const;
    std::map<KernelId, DimensionVector> getModifiedLocalSizes(const std::vector<ParameterPair>& parameterPairs) const;
    std::map<KernelId, std::vector<LocalMemoryModifier>> getLocalMemoryModifiers(const std::vector<ParameterPair>& parameterPairs) const;

    // Getters
    KernelId getId() const;
    std::string getName() const;
    std::vector<const Kernel*> getKernels() const;
    std::vector<KernelParameter> getParameters() const;
    std::vector<KernelConstraint> getConstraints() const;
    std::vector<KernelParameterPack> getParameterPacks() const;
    std::vector<ArgumentId> getSharedArgumentIds() const;
    std::vector<ArgumentId> getKernelArgumentIds(const KernelId id) const;
    bool hasParameter(const std::string& parameterName) const;

private:
    // Attributes
    KernelId id;
    std::string name;
    std::vector<const Kernel*> kernels;
    std::vector<KernelParameter> parameters;
    std::vector<KernelConstraint> constraints;
    std::vector<KernelParameterPack> parameterPacks;
    std::vector<ArgumentId> sharedArgumentIds;
    std::map<KernelId, std::vector<ArgumentId>> kernelArgumentIds;
    std::map<KernelId, std::array<std::vector<std::string>, 3>> globalThreadModifierNames;
    std::map<KernelId, std::array<std::function<size_t(const size_t, const std::vector<size_t>&)>, 3>> globalThreadModifiers;
    std::map<KernelId, std::array<std::vector<std::string>, 3>> localThreadModifierNames;
    std::map<KernelId, std::array<std::function<size_t(const size_t, const std::vector<size_t>&)>, 3>> localThreadModifiers;
    std::map<KernelId, std::map<ArgumentId, std::vector<std::string>>> localMemoryModifierNames;
    std::map<KernelId, std::map<ArgumentId, std::function<size_t(const size_t, const std::vector<size_t>&)>>> localMemoryModifiers;

    void validateModifierParameters(const std::vector<std::string>& parameterNames) const;
};

} // namespace ktt
