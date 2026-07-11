
#pragma once

#include "config.hpp"
#include <vector>
#include <memory>



class VectorImpl {
public:
    virtual ~VectorImpl() = default;
    
    virtual size_t size() const = 0;
    virtual float get_element(size_t index) const = 0;
    virtual void set_element(size_t index, float value) = 0;
    
    virtual void add(const VectorImpl& other, VectorImpl& result) const = 0;
    virtual void subtract(const VectorImpl& other, VectorImpl& result) const = 0;
    virtual void multiply(float scalar, VectorImpl& result) const = 0;
    virtual void add_assign(const VectorImpl& other) = 0;
    virtual void subtract_assign(const VectorImpl& other) = 0;
    virtual void multiply_assign(float scalar) = 0;
    
    virtual float dot(const VectorImpl& other) const = 0;
    virtual float sum() const = 0;
    virtual float mean() const = 0;
    virtual float norm() const = 0;
    virtual void zero() = 0;
};


class VectorCPU : public VectorImpl {
private:
    std::vector<float> data_;

public:
    explicit VectorCPU(size_t size);
    
    size_t size() const override;
    
    float get_element(size_t index) const override;
    void set_element(size_t index, float value) override;
    
    void add(const VectorImpl& other, VectorImpl& result) const override;
    void subtract(const VectorImpl& other, VectorImpl& result) const override;
    void multiply(float scalar, VectorImpl& result) const override;
    
    void add_assign(const VectorImpl& other) override;
    void subtract_assign(const VectorImpl& other) override;
    void multiply_assign(float scalar) override;
    
    float dot(const VectorImpl& other) const override;
    float sum() const override;
    float mean() const override;
    float norm() const override;
    void zero() override;
};


class VectorGPU : public VectorImpl {
private:
    size_t size_;
    float* data_;            // сырой указатель — имитация памяти устройства

public:
    explicit VectorGPU(size_t size);
    ~VectorGPU();
    
    // Запрет копирования (пока, чтобы не усложнять)
    VectorGPU(const VectorGPU&) = delete;
    VectorGPU& operator=(const VectorGPU&) = delete;
    
    size_t size() const override;
    
    float get_element(size_t index) const override;
    void set_element(size_t index, float value) override;
    
    void add(const VectorImpl& other, VectorImpl& result) const override;
    void subtract(const VectorImpl& other, VectorImpl& result) const override;
    void multiply(float scalar, VectorImpl& result) const override;
    
    void add_assign(const VectorImpl& other) override;
    void subtract_assign(const VectorImpl& other) override;
    void multiply_assign(float scalar) override;
    
    float dot(const VectorImpl& other) const override;
    float sum() const override;
    float mean() const override;
    float norm() const override;
    void zero() override;
};


class Vector {
private:
    Device device_;
    std::unique_ptr<VectorImpl> impl_;
    
    static std::unique_ptr<VectorImpl> create_impl(size_t size, Device device);

public:
    Vector();
    explicit Vector(size_t size, Device device = Device::CPU); 

    size_t size() const;
    
    float at(size_t index) const;
    void set_at(size_t index, float value);
    
    Vector operator+(const Vector& other) const;
    Vector operator-(const Vector& other) const;
    Vector operator*(float scalar) const;
    
    Vector& operator+=(const Vector& other);
    Vector& operator-=(const Vector& other);
    Vector& operator*=(float scalar);
    
    float dot(const Vector& other) const;
    float sum() const;
    float mean() const;
    float norm() const;
    void zero();
};

Vector operator*(float scalar, const Vector& vector);

