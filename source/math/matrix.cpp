#include "math.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>


Matrix::Matrix()
    : rows_(0),
      cols_(0)
{
}


Matrix::Matrix(size_t rows, size_t cols)
    : rows_(rows),
      cols_(cols),
      data_(rows * cols, 0.0f)
{
}


size_t Matrix::rows() const
{
    return rows_;
}


size_t Matrix::cols() const
{
    return cols_;
}


size_t Matrix::size() const
{
    return data_.size();
}


float& Matrix::operator()(size_t row, size_t col)
{
    if (row >= rows_ || col >= cols_)
    {
        throw std::out_of_range("Matrix index out of range");
    }

    return data_[row * cols_ + col];
}


const float& Matrix::operator()(size_t row, size_t col) const
{
    if (row >= rows_ || col >= cols_)
    {
        throw std::out_of_range("Matrix index out of range");
    }

    return data_[row * cols_ + col];
}


Matrix Matrix::transpose() const
{
    Matrix result(cols_, rows_);


    for (size_t i = 0; i < rows_; i++)
    {
        for (size_t j = 0; j < cols_; j++)
        {
            result(j, i) = (*this)(i, j);
        }
    }


    return result;
}


Matrix Matrix::matmul(const Matrix& other) const
{
    if (cols_ != other.rows_)
    {
        throw std::invalid_argument(
            "Matrix dimensions mismatch for multiplication"
        );
    }


    Matrix result(rows_, other.cols_);


    for (size_t i = 0; i < rows_; i++)
    {
        for (size_t j = 0; j < other.cols_; j++)
        {
            float sum = 0.0f;


            for (size_t k = 0; k < cols_; k++)
            {
                sum += (*this)(i, k) * other(k, j);
            }


            result(i, j) = sum;
        }
    }


    return result;
}


Matrix Matrix::operator+(const Matrix& other) const
{
    if (rows_ != other.rows_ ||
        cols_ != other.cols_)
    {
        throw std::invalid_argument(
            "Matrix dimensions mismatch for addition"
        );
    }


    Matrix result(rows_, cols_);


    for (size_t i = 0; i < data_.size(); i++)
    {
        result.data_[i] = data_[i] + other.data_[i];
    }


    return result;
}


Matrix Matrix::operator*(float value) const
{
    Matrix result(rows_, cols_);


    for (size_t i = 0; i < data_.size(); i++)
    {
        result.data_[i] = data_[i] * value;
    }


    return result;
}


void Matrix::zero()
{
    std::fill(
        data_.begin(),
        data_.end(),
        0.0f
    );
}


void Matrix::fill_random(float min, float max)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist(min, max);


    for (float& value : data_)
    {
        value = dist(gen);
    }
}

