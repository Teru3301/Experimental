
#include "math.hpp"
#include <stdexcept>



std::unique_ptr<VectorImpl> Vector::create_impl(size_t size, Device device) {
    switch (device) {
        case Device::CPU:
            return std::make_unique<VectorCPU>(size);
        case Device::GPU:
            return std::make_unique<VectorGPU>(size);
        default:
            throw std::invalid_argument("Неподдерживаемое устройство");
    }
}


Vector::Vector() : device_(Device::CPU), impl_(std::make_unique<VectorCPU>(0)) {}
Vector::Vector(size_t size, Device device) 
    : device_(device), impl_(create_impl(size, device)) {}


size_t Vector::size() const 
{
    return impl_->size();
}


float& Vector::operator[](size_t index)
{
    if (index >= size()) {
        throw std::out_of_range("Индекс выходит за пределы вектора");
    }
    return impl_->element(index);
}
    

const float& Vector::operator[](size_t index) const
{
    if (index >= size()) {
        throw std::out_of_range("Индекс выходит за пределы вектора");
    }
    return impl_->element(index);
}


Vector Vector::operator+(const Vector& other) const
{
    if (device_ != other.device_)
    {
        throw std::invalid_argument("Нельзя выполнять операции над векторами с разных устройств");
    }
    if (size() != other.size()) {
        throw std::invalid_argument("Векторы должны иметь одинаковый размер");
    }

    Vector result(size(), device_);
    impl_->add(*other.impl_, *result.impl_);
    return result;
}
    

Vector Vector::operator-(const Vector& other) const
{
    if (device_ != other.device_)
    {
        throw std::invalid_argument("Нельзя выполнять операции над векторами с разных устройств");
    }
    if (size() != other.size()) {
        throw std::invalid_argument("Векторы должны иметь одинаковый размер");
    }

    Vector result(size(), device_);
    impl_->subtract(*other.impl_, *result.impl_);
    return result;
}


Vector Vector::operator*(float scalar) const
{
    Vector result(size(), device_);
    impl_->multiply(scalar, *result.impl_);
    return result;
}


Vector& Vector::operator+=(const Vector& other) 
{
    if (device_ != other.device_)
    {
        throw std::invalid_argument("Нельзя выполнять операции над векторами с разных устройств");
    }
    if (size() != other.size()) {
        throw std::invalid_argument("Векторы должны иметь одинаковый размер");
    }

    impl_->add_assign(*other.impl_);
    return *this;
}

Vector& Vector::operator-=(const Vector& other)
{
    if (device_ != other.device_)
    {
        throw std::invalid_argument("Нельзя выполнять операции над векторами с разных устройств");
    }
    if (size() != other.size()) {
        throw std::invalid_argument("Векторы должны иметь одинаковый размер");
    }

    impl_->subtract_assign(*other.impl_);
    return *this;
}


Vector& Vector::operator*=(float scalar)
{
    impl_->multiply_assign(scalar);
    return *this;
}


float Vector::dot(const Vector& other) const 
{
    if (device_ != other.device_)
    {
        throw std::invalid_argument("Нельзя выполнять операции над векторами с разных устройств");
    }
    if (size() != other.size()) {
        throw std::invalid_argument("Векторы должны иметь одинаковый размер для скалярного произведения");
    }

    return impl_->dot(*other.impl_);
}


float Vector::sum() const
{
    return impl_->sum();
}


float Vector::mean() const
{
    return impl_->mean();
}


float Vector::norm() const
{
    return impl_->norm();
}


void Vector::zero()
{
    impl_->zero();
}


Vector operator*(float scalar, const Vector& vector) {
    return vector * scalar;
}

