#include "math.hpp"
#include <cmath>

VectorGPU::VectorGPU(size_t size) : data_(size, 0.0f) {}

size_t VectorGPU::size() const {
    return data_.size();
}

float& VectorGPU::element(size_t index) {
    return data_[index];
}

const float& VectorGPU::element(size_t index) const {
    return data_[index];
}

void VectorGPU::add(const VectorImpl& other, VectorImpl& result) const {
    const auto& other_gpu = static_cast<const VectorGPU&>(other);
    auto& result_gpu = static_cast<VectorGPU&>(result);
    
    // В реальном GPU коде здесь был бы запуск CUDA ядра
    for (size_t i = 0; i < data_.size(); ++i) {
        result_gpu.data_[i] = data_[i] + other_gpu.data_[i];
    }
}

void VectorGPU::subtract(const VectorImpl& other, VectorImpl& result) const {
    const auto& other_gpu = static_cast<const VectorGPU&>(other);
    auto& result_gpu = static_cast<VectorGPU&>(result);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        result_gpu.data_[i] = data_[i] - other_gpu.data_[i];
    }
}

void VectorGPU::multiply(float scalar, VectorImpl& result) const {
    auto& result_gpu = static_cast<VectorGPU&>(result);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        result_gpu.data_[i] = data_[i] * scalar;
    }
}

void VectorGPU::add_assign(const VectorImpl& other) {
    const auto& other_gpu = static_cast<const VectorGPU&>(other);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other_gpu.data_[i];
    }
}

void VectorGPU::subtract_assign(const VectorImpl& other) {
    const auto& other_gpu = static_cast<const VectorGPU&>(other);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] -= other_gpu.data_[i];
    }
}

void VectorGPU::multiply_assign(float scalar) {
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] *= scalar;
    }
}

float VectorGPU::dot(const VectorImpl& other) const {
    const auto& other_gpu = static_cast<const VectorGPU&>(other);
    float result = 0.0f;
    
    // В реальности здесь была бы параллельная редукция на GPU
    for (size_t i = 0; i < data_.size(); ++i) {
        result += data_[i] * other_gpu.data_[i];
    }
    
    return result;
}

float VectorGPU::sum() const {
    float result = 0.0f;
    
    for (size_t i = 0; i < data_.size(); ++i) {
        result += data_[i];
    }
    
    return result;
}

float VectorGPU::mean() const {
    return data_.empty() ? 0.0f : sum() / data_.size();
}

float VectorGPU::norm() const {
    return std::sqrt(dot(*this));
}

void VectorGPU::zero() {
    std::fill(data_.begin(), data_.end(), 0.0f);
}

