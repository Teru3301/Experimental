
#include "math/vector.hpp"
#include <cmath>
#include <algorithm>



VectorCPU::VectorCPU(size_t size) : data_(size, 0.0f) {}


size_t VectorCPU::size() const 
{
    return data_.size();
}


float VectorCPU::get_element(size_t index) const
{
    return data_[index];
}


void VectorCPU::set_element(size_t index, float value)
{
    data_[index] = value;
}


void VectorCPU::add(const VectorImpl& other, VectorImpl& result) const 
{
    const auto& other_cpu = static_cast<const VectorCPU&>(other);
    auto& result_cpu = static_cast<VectorCPU&>(result);

    for (size_t i = 0; i < data_.size(); ++i) 
    {
        result_cpu.data_[i] = data_[i] + other_cpu.data_[i];
    }
}


void VectorCPU::subtract(const VectorImpl& other, VectorImpl& result) const 
{
    const auto& other_cpu = static_cast<const VectorCPU&>(other);
    auto& result_cpu = static_cast<VectorCPU&>(result);

    for (size_t i = 0; i < data_.size(); ++i) 
    {
        result_cpu.data_[i] = data_[i] - other_cpu.data_[i];
    }
}


void VectorCPU::multiply(float scalar, VectorImpl& result) const 
{
    auto& result_cpu = static_cast<VectorCPU&>(result);

    for (size_t i = 0; i < data_.size(); ++i) 
    {
        result_cpu.data_[i] = data_[i] * scalar;
    }
}


void VectorCPU::add_assign(const VectorImpl& other) 
{
    const auto& other_cpu = static_cast<const VectorCPU&>(other);

    for (size_t i = 0; i < data_.size(); ++i) 
    {
        data_[i] += other_cpu.data_[i];
    }
}


void VectorCPU::subtract_assign(const VectorImpl& other) 
{
    const auto& other_cpu = static_cast<const VectorCPU&>(other);

    for (size_t i = 0; i < data_.size(); ++i) 
    {
        data_[i] -= other_cpu.data_[i];
    }
}


void VectorCPU::multiply_assign(float scalar) 
{
    for (size_t i = 0; i < data_.size(); ++i) 
    {
        data_[i] *= scalar;
    }
}


float VectorCPU::dot(const VectorImpl& other) const 
{
    const auto& other_cpu = static_cast<const VectorCPU&>(other);
    float result = 0.0f;

    for (size_t i = 0; i < data_.size(); ++i) 
    {
        result += data_[i] * other_cpu.data_[i];
    }

    return result;
}


float VectorCPU::sum() const 
{
    float result = 0.0f;

    for (size_t i = 0; i < data_.size(); ++i) 
    {
        result += data_[i];
    }

    return result;
}


float VectorCPU::mean() const 
{
    return data_.empty() ? 0.0f : sum() / data_.size();
}


float VectorCPU::norm() const 
{
    return std::sqrt(dot(*this));
}


void VectorCPU::zero() 
{
    std::fill(data_.begin(), data_.end(), 0.0f);
}

