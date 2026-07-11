
#include "math/vector.hpp"
#include <cmath>
#include <cstring>



VectorGPU::VectorGPU(size_t size) : size_(size)
{
    data_ = new float[size]();  // () зануляет память
}


VectorGPU::~VectorGPU()
{
    delete[] data_;
}


size_t VectorGPU::size() const
{
    return size_;
}


float VectorGPU::get_element(size_t index) const
{
    return data_[index];
}


void VectorGPU::set_element(size_t index, float value)
{
    data_[index] = value;
}


void VectorGPU::add(const VectorImpl& other, VectorImpl& result) const
{
    const auto& other_gpu = static_cast<const VectorGPU&>(other);
    auto& result_gpu = static_cast<VectorGPU&>(result);

    // ЗАГЛУШКА: позже заменится на SYCL-ядро
    for (size_t i = 0; i < size_; ++i)
    {
        result_gpu.data_[i] = data_[i] + other_gpu.data_[i];
    }
}


void VectorGPU::subtract(const VectorImpl& other, VectorImpl& result) const
{
    const auto& other_gpu = static_cast<const VectorGPU&>(other);
    auto& result_gpu = static_cast<VectorGPU&>(result);

    // ЗАГЛУШКА
    for (size_t i = 0; i < size_; ++i)
    {
        result_gpu.data_[i] = data_[i] - other_gpu.data_[i];
    }
}


void VectorGPU::multiply(float scalar, VectorImpl& result) const
{
    auto& result_gpu = static_cast<VectorGPU&>(result);

    // ЗАГЛУШКА
    for (size_t i = 0; i < size_; ++i)
    {
        result_gpu.data_[i] = data_[i] * scalar;
    }
}


void VectorGPU::add_assign(const VectorImpl& other)
{
    const auto& other_gpu = static_cast<const VectorGPU&>(other);

    // ЗАГЛУШКА
    for (size_t i = 0; i < size_; ++i)
    {
        data_[i] += other_gpu.data_[i];
    }
}


void VectorGPU::subtract_assign(const VectorImpl& other)
{
    const auto& other_gpu = static_cast<const VectorGPU&>(other);

    // ЗАГЛУШКА
    for (size_t i = 0; i < size_; ++i)
    {
        data_[i] -= other_gpu.data_[i];
    }
}


void VectorGPU::multiply_assign(float scalar)
{
    // ЗАГЛУШКА
    for (size_t i = 0; i < size_; ++i)
    {
        data_[i] *= scalar;
    }
}


float VectorGPU::dot(const VectorImpl& other) const
{
    const auto& other_gpu = static_cast<const VectorGPU&>(other);
    float result = 0.0f;

    // ЗАГЛУШКА: позже — параллельная редукция на GPU
    for (size_t i = 0; i < size_; ++i)
    {
        result += data_[i] * other_gpu.data_[i];
    }

    return result;
}


float VectorGPU::sum() const
{
    float result = 0.0f;

    // ЗАГЛУШКА
    for (size_t i = 0; i < size_; ++i)
    {
        result += data_[i];
    }

    return result;
}


float VectorGPU::mean() const
{
    return size_ == 0 ? 0.0f : sum() / size_;
}


float VectorGPU::norm() const
{
    return std::sqrt(dot(*this));
}


void VectorGPU::zero()
{
    std::memset(data_, 0, size_ * sizeof(float));
}

