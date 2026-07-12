
#include "math/vector.hpp"
#include <cmath>
#include <cstring>
#include "device/queue.hpp"



VectorGPU::VectorGPU(size_t size) : size_(size)
{
    sycl::queue& q = get_default_queue();
    data_ = sycl::malloc_device<float>(size_, q);
    q.memset(data_, 0, size_ * sizeof(float)).wait();
}


VectorGPU::~VectorGPU()
{
    sycl::queue& q = get_default_queue();
    sycl::free(data_, q);
}


size_t VectorGPU::size() const
{
    return size_;
}


float VectorGPU::get_element(size_t index) const
{
    sycl::queue& q = get_default_queue();
    float value;
    q.memcpy(&value, data_ + index, sizeof(float)).wait();
    return value;
}


void VectorGPU::set_element(size_t index, float value)
{
    sycl::queue& q = get_default_queue();
    q.memcpy(data_ + index, &value, sizeof(float)).wait();
}


void VectorGPU::add(const VectorImpl& other, VectorImpl& result) const
{
    const auto& other_gpu = static_cast<const VectorGPU&>(other);
    auto& result_gpu = static_cast<VectorGPU&>(result);
    sycl::queue& q = get_default_queue();

    float* my_data = data_;
    float* other_data = other_gpu.data_;
    float* result_data = result_gpu.data_;

    q.parallel_for(sycl::range<1>(size_), [=](sycl::item<1> it) {
        size_t i = it.get_id(0);
        result_data[i] = my_data[i] + other_data[i];
    }).wait();
}


void VectorGPU::subtract(const VectorImpl& other, VectorImpl& result) const
{
    const auto& other_gpu = static_cast<const VectorGPU&>(other);
    auto& result_gpu = static_cast<VectorGPU&>(result);
    sycl::queue& q = get_default_queue();

    float* my_data = data_;
    float* other_data = other_gpu.data_;
    float* result_data = result_gpu.data_;

    q.parallel_for(sycl::range<1>(size_), [=](sycl::item<1> it) {
        size_t i = it.get_id(0);
        result_data[i] = my_data[i] - other_data[i];
    }).wait();
}


void VectorGPU::multiply(float scalar, VectorImpl& result) const
{
    auto& result_gpu = static_cast<VectorGPU&>(result);
    sycl::queue& q = get_default_queue();

    float* my_data = data_;
    float* result_data = result_gpu.data_;

    q.parallel_for(sycl::range<1>(size_), [=](sycl::item<1> it) {
        size_t i = it.get_id(0);
        result_data[i] = my_data[i] * scalar;
    }).wait();
}


void VectorGPU::add_assign(const VectorImpl& other)
{
    const auto& other_gpu = static_cast<const VectorGPU&>(other);
    sycl::queue& q = get_default_queue();

    float* my_data = data_;
    float* other_data = other_gpu.data_;

    q.parallel_for(sycl::range<1>(size_), [=](sycl::item<1> it) {
        size_t i = it.get_id(0);
        my_data[i] += other_data[i];
    }).wait();
}


void VectorGPU::subtract_assign(const VectorImpl& other)
{
    const auto& other_gpu = static_cast<const VectorGPU&>(other);
    sycl::queue& q = get_default_queue();

    float* my_data = data_;
    float* other_data = other_gpu.data_;

    q.parallel_for(sycl::range<1>(size_), [=](sycl::item<1> it) {
        size_t i = it.get_id(0);
        my_data[i] -= other_data[i];
    }).wait();
}


void VectorGPU::multiply_assign(float scalar)
{
    sycl::queue& q = get_default_queue();
    float* my_data = data_;

    q.parallel_for(sycl::range<1>(size_), [=](sycl::item<1> it) {
        size_t i = it.get_id(0);
        my_data[i] *= scalar;
    }).wait();
}


float VectorGPU::dot(const VectorImpl& other) const
{
    const auto& other_gpu = static_cast<const VectorGPU&>(other);
    sycl::queue& q = get_default_queue();

    float* my_data = data_;
    float* other_data = other_gpu.data_;

    // Выделяем память под результат на GPU
    float* d_result = sycl::malloc_device<float>(1, q);
    q.memset(d_result, 0, sizeof(float)).wait();

    q.parallel_for(sycl::range<1>(size_), [=](sycl::item<1> it) {
        size_t i = it.get_id(0);
        float product = my_data[i] * other_data[i];
        // Атомарное сложение — потоки не мешают друг другу
        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
            atomic_result(d_result[0]);
        atomic_result.fetch_add(product);
    }).wait();

    // Копируем результат с GPU на CPU
    float result;
    q.memcpy(&result, d_result, sizeof(float)).wait();
    sycl::free(d_result, q);
    return result;
}


float VectorGPU::sum() const
{
    sycl::queue& q = get_default_queue();
    float* my_data = data_;

    float* d_result = sycl::malloc_device<float>(1, q);
    q.memset(d_result, 0, sizeof(float)).wait();

    q.parallel_for(sycl::range<1>(size_), [=](sycl::item<1> it) {
        size_t i = it.get_id(0);
        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
            atomic_result(d_result[0]);
        atomic_result.fetch_add(my_data[i]);
    }).wait();

    float result;
    q.memcpy(&result, d_result, sizeof(float)).wait();
    sycl::free(d_result, q);
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
    sycl::queue& q = get_default_queue();
    float* my_data = data_;

    q.parallel_for(sycl::range<1>(size_), [=](sycl::item<1> it) {
        size_t i = it.get_id(0);
        my_data[i] = 0.0f;
    }).wait();
}

