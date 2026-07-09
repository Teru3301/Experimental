
#pragma once

#include "math/vector.hpp"
#include <vector>
#include <span>



class Matrix
{
private:
    size_t rows_;
    size_t cols_;
    std::vector<float> data_;

public:
    Matrix();
    Matrix(size_t rows, size_t cols);

    size_t rows() const;
    size_t cols() const;
    size_t size() const;

    float& operator()(size_t row, size_t col);
    const float& operator()(size_t row, size_t col) const;

    Matrix transpose() const;

    Matrix matmul(const Matrix& other) const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator*(float value) const;

    void zero();
    void fill_random(float min, float max);

};


class Tensor
{
private:
    std::vector<float> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

    size_t offset(std::span<const size_t> position) const;

public:
    Tensor();
    explicit Tensor(std::span<const size_t> shape);

    size_t dimensions() const;
    size_t size() const;

    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides() const;

    float* data();
    const float* data() const;

    float& operator()(std::span<const size_t>);
    const float& operator()(std::span<const size_t>) const;

    Tensor reshape(std::span<const size_t>) const;
    Tensor transpose(size_t dim1, size_t dim2) const;
    void fill_random(float min, float max);

    Tensor softmax(size_t dimension);

    Tensor sum(size_t dimension);

    Tensor mean(size_t dimension);

};


