#include <cstring>
#include <stdexcept>
#include "kernel_argument.h"

namespace ktt
{

KernelArgument::KernelArgument(const ArgumentId id, const size_t numberOfElements, const ArgumentDataType& dataType,
    const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType, const ArgumentUploadType& uploadType) :
    id(id),
    numberOfElements(numberOfElements),
    argumentDataType(dataType),
    argumentMemoryLocation(memoryLocation),
    argumentAccessType(accessType),
    argumentUploadType(uploadType),
    dataOwned(true)
{
    if (numberOfElements == 0)
    {
        throw std::runtime_error("Data provided for kernel argument is empty");
    }
    prepareData(numberOfElements, argumentDataType);
}

KernelArgument::KernelArgument(const ArgumentId id, const void* data, const size_t numberOfElements, const ArgumentDataType& dataType,
    const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType, const ArgumentUploadType& uploadType) :
    KernelArgument(id, data, numberOfElements, dataType, memoryLocation, accessType, uploadType, true)
{}

KernelArgument::KernelArgument(const ArgumentId id, const void* data, const size_t numberOfElements, const ArgumentDataType& dataType,
    const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType, const ArgumentUploadType& uploadType, const bool dataOwned) :
    id(id),
    numberOfElements(numberOfElements),
    argumentDataType(dataType),
    argumentMemoryLocation(memoryLocation),
    argumentAccessType(accessType),
    argumentUploadType(uploadType),
    dataOwned(dataOwned)
{
    if (numberOfElements == 0)
    {
        throw std::runtime_error("Data provided for kernel argument is empty");
    }

    if (dataOwned && data != nullptr)
    {
        initializeData(data, numberOfElements, argumentDataType);
    }
    if (!dataOwned)
    {
        referencedData = data;
    }
}

void KernelArgument::updateData(const void* data, const size_t numberOfElements)
{
    if (numberOfElements == 0)
    {
        throw std::runtime_error("Data provided for kernel argument is empty");
    }

    this->numberOfElements = numberOfElements;
    if (dataOwned)
    {
        initializeData(data, numberOfElements, argumentDataType);
    }
    else
    {
        referencedData = data;
    }
}

ArgumentId KernelArgument::getId() const
{
    return id;
}

size_t KernelArgument::getNumberOfElements() const
{
    return numberOfElements;
}

ArgumentDataType KernelArgument::getDataType() const
{
    return argumentDataType;
}

ArgumentMemoryLocation KernelArgument::getMemoryLocation() const
{
    return argumentMemoryLocation;
}

ArgumentAccessType KernelArgument::getAccessType() const
{
    return argumentAccessType;
}

ArgumentUploadType KernelArgument::getUploadType() const
{
    return argumentUploadType;
}

size_t KernelArgument::getElementSizeInBytes() const
{
    switch (argumentDataType)
    {
    case ArgumentDataType::Char:
        return sizeof(int8_t);
    case ArgumentDataType::UnsignedChar:
        return sizeof(uint8_t);
    case ArgumentDataType::Short:
        return sizeof(int16_t);
    case ArgumentDataType::UnsignedShort:
        return sizeof(uint16_t);
    case ArgumentDataType::Int:
        return sizeof(int32_t);
    case ArgumentDataType::UnsignedInt:
        return sizeof(uint32_t);
    case ArgumentDataType::Long:
        return sizeof(int64_t);
    case ArgumentDataType::UnsignedLong:
        return sizeof(uint64_t);
    case ArgumentDataType::Half:
        return sizeof(half);
    case ArgumentDataType::Float:
        return sizeof(float);
    case ArgumentDataType::Double:
        return sizeof(double);
    default:
        throw std::runtime_error("Unsupported argument data type");
    }
}

size_t KernelArgument::getDataSizeInBytes() const
{
    return numberOfElements * getElementSizeInBytes();
}

const void* KernelArgument::getData() const
{
    if (!dataOwned)
    {
        return referencedData;
    }

    switch (argumentDataType)
    {
    case ArgumentDataType::Char:
        return (void*)dataChar.data();
    case ArgumentDataType::UnsignedChar:
        return (void*)dataUnsignedChar.data();
    case ArgumentDataType::Short:
        return (void*)dataShort.data();
    case ArgumentDataType::UnsignedShort:
        return (void*)dataUnsignedShort.data();
    case ArgumentDataType::Int:
        return (void*)dataInt.data();
    case ArgumentDataType::UnsignedInt:
        return (void*)dataUnsignedInt.data();
    case ArgumentDataType::Long:
        return (void*)dataLong.data();
    case ArgumentDataType::UnsignedLong:
        return (void*)dataUnsignedLong.data();
    case ArgumentDataType::Half:
        return (void*)dataHalf.data();
    case ArgumentDataType::Float:
        return (void*)dataFloat.data();
    case ArgumentDataType::Double:
        return (void*)dataDouble.data();
    default:
        throw std::runtime_error("Unsupported argument data type");
    }
}

void* KernelArgument::getData()
{
    return const_cast<void*>(static_cast<const KernelArgument*>(this)->getData());
}

std::vector<int8_t> KernelArgument::getDataChar() const
{
    return dataChar;
}

std::vector<uint8_t> KernelArgument::getDataUnsignedChar() const
{
    return dataUnsignedChar;
}

std::vector<int16_t> KernelArgument::getDataShort() const
{
    return dataShort;
}

std::vector<uint16_t> KernelArgument::getDataUnsignedShort() const
{
    return dataUnsignedShort;
}

std::vector<int32_t> KernelArgument::getDataInt() const
{
    return dataInt;
}

std::vector<uint32_t> KernelArgument::getDataUnsignedInt() const
{
    return dataUnsignedInt;
}

std::vector<int64_t> KernelArgument::getDataLong() const
{
    return dataLong;
}

std::vector<uint64_t> KernelArgument::getDataUnsignedLong() const
{
    return dataUnsignedLong;
}

std::vector<half> KernelArgument::getDataHalf() const
{
    return dataHalf;
}

std::vector<float> KernelArgument::getDataFloat() const
{
    return dataFloat;
}

std::vector<double> KernelArgument::getDataDouble() const
{
    return dataDouble;
}

bool KernelArgument::operator==(const KernelArgument& other) const
{
    return id == other.id;
}

bool KernelArgument::operator!=(const KernelArgument& other) const
{
    return !(*this == other);
}

void KernelArgument::initializeData(const void* data, const size_t numberOfElements, const ArgumentDataType& dataType)
{
    prepareData(numberOfElements, dataType);
    std::memcpy(getData(), data, numberOfElements * getElementSizeInBytes());
}

void KernelArgument::prepareData(const size_t numberOfElements, const ArgumentDataType& dataType)
{
    if (dataType == ArgumentDataType::Char)
    {
        dataChar.resize(numberOfElements);
    }
    else if (dataType == ArgumentDataType::UnsignedChar)
    {
        dataUnsignedChar.resize(numberOfElements);
    }
    else if (dataType == ArgumentDataType::Short)
    {
        dataShort.resize(numberOfElements);
    }
    else if (dataType == ArgumentDataType::UnsignedShort)
    {
        dataUnsignedShort.resize(numberOfElements);
    }
    else if (dataType == ArgumentDataType::Int)
    {
        dataInt.resize(numberOfElements);
    }
    else if (dataType == ArgumentDataType::UnsignedInt)
    {
        dataUnsignedInt.resize(numberOfElements);
    }
    else if (dataType == ArgumentDataType::Long)
    {
        dataLong.resize(numberOfElements);
    }
    else if (dataType == ArgumentDataType::UnsignedLong)
    {
        dataUnsignedLong.resize(numberOfElements);
    }
    else if (dataType == ArgumentDataType::Half)
    {
        dataHalf.resize(numberOfElements);
    }
    else if (dataType == ArgumentDataType::Float)
    {
        dataFloat.resize(numberOfElements);
    }
    else if (dataType == ArgumentDataType::Double)
    {
        dataDouble.resize(numberOfElements);
    }
    else
    {
        throw std::runtime_error("Unsupported argument data type was provided for kernel argument");
    }
}

} // namespace ktt
