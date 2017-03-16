#pragma once

#include <algorithm>
#include <random>

#include "../kernel/kernel_configuration.h"
#include "searcher.h"

namespace ktt
{

class RandomSearcher : public Searcher
{
public:
    RandomSearcher(const std::vector<KernelConfiguration>& configurations, const double fraction):
        configurations(configurations),
        index(0),
        fraction(fraction)
    {
        if (configurations.size() == 0)
        {
            throw std::runtime_error("Configurations vector provided for searcher is empty");
        }
        auto engine = std::default_random_engine();
        std::shuffle(std::begin(this->configurations), std::end(this->configurations), engine);
    }

    virtual KernelConfiguration getNextConfiguration() override
    {
        return configurations.at(index);
    }

    virtual void calculateNextConfiguration(const double previousConfigurationDuration) override
    {
        if (index < getConfigurationsCount() - 1)
        {
            index++;
        }
    }

    virtual size_t getConfigurationsCount() override
    {
        return std::max(static_cast<size_t>(1), static_cast<size_t>(configurations.size() * fraction));
    }

private:
    std::vector<KernelConfiguration> configurations;
    size_t index;
    double fraction;
};

} // namespace ktt
