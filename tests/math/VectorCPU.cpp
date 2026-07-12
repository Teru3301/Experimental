#include <gtest/gtest.h>
#include "math.hpp"

TEST(VectorCPUTest, DefaultConstructor)
{
    Vector v;
    EXPECT_EQ(v.size(), 0);
}

TEST(VectorCPUTest, Constructor)
{
    Vector v(10, Device::CPU);
    EXPECT_EQ(v.size(), 10);
}

TEST(VectorCPUTest, ElementAccess)
{
    Vector v(3, Device::CPU);
    v.set_at(0, 1.0f);
    v.set_at(1, 2.0f);
    v.set_at(2, 3.0f);
    EXPECT_FLOAT_EQ(v.at(0), 1.0f);
    EXPECT_FLOAT_EQ(v.at(1), 2.0f);
    EXPECT_FLOAT_EQ(v.at(2), 3.0f);
}

TEST(VectorCPUTest, Addition)
{
    Vector a(3, Device::CPU);
    Vector b(3, Device::CPU);
    a.set_at(0, 1.0f); a.set_at(1, 2.0f); a.set_at(2, 3.0f);
    b.set_at(0, 4.0f); b.set_at(1, 5.0f); b.set_at(2, 6.0f);
    Vector c = a + b;
    EXPECT_FLOAT_EQ(c.at(0), 5.0f);
    EXPECT_FLOAT_EQ(c.at(1), 7.0f);
    EXPECT_FLOAT_EQ(c.at(2), 9.0f);
}

TEST(VectorCPUTest, Subtraction)
{
    Vector a(3, Device::CPU);
    Vector b(3, Device::CPU);
    a.set_at(0, 5.0f); a.set_at(1, 7.0f); a.set_at(2, 9.0f);
    b.set_at(0, 1.0f); b.set_at(1, 2.0f); b.set_at(2, 3.0f);
    Vector c = a - b;
    EXPECT_FLOAT_EQ(c.at(0), 4.0f);
    EXPECT_FLOAT_EQ(c.at(1), 5.0f);
    EXPECT_FLOAT_EQ(c.at(2), 6.0f);
}

TEST(VectorCPUTest, ScalarMultiplication)
{
    Vector v(3, Device::CPU);
    v.set_at(0, 1.0f); v.set_at(1, 2.0f); v.set_at(2, 3.0f);
    Vector result = v * 2.0f;
    EXPECT_FLOAT_EQ(result.at(0), 2.0f);
    EXPECT_FLOAT_EQ(result.at(1), 4.0f);
    EXPECT_FLOAT_EQ(result.at(2), 6.0f);
}

TEST(VectorCPUTest, DotProduct)
{
    Vector a(3, Device::CPU);
    Vector b(3, Device::CPU);
    a.set_at(0, 1.0f); a.set_at(1, 2.0f); a.set_at(2, 3.0f);
    b.set_at(0, 4.0f); b.set_at(1, 5.0f); b.set_at(2, 6.0f);
    EXPECT_FLOAT_EQ(a.dot(b), 32.0f);
}

TEST(VectorCPUTest, SumAndMean)
{
    Vector v(4, Device::CPU);
    v.set_at(0, 1.0f); v.set_at(1, 2.0f); v.set_at(2, 3.0f); v.set_at(3, 4.0f);
    EXPECT_FLOAT_EQ(v.sum(), 10.0f);
    EXPECT_FLOAT_EQ(v.mean(), 2.5f);
}

TEST(VectorCPUTest, Norm)
{
    Vector v(3, Device::CPU);
    v.set_at(0, 3.0f); v.set_at(1, 4.0f); v.set_at(2, 0.0f);
    EXPECT_FLOAT_EQ(v.norm(), 5.0f);
}

TEST(VectorCPUTest, Zero)
{
    Vector v(3, Device::CPU);
    v.set_at(0, 1.0f); v.set_at(1, 2.0f); v.set_at(2, 3.0f);
    v.zero();
    EXPECT_FLOAT_EQ(v.at(0), 0.0f);
    EXPECT_FLOAT_EQ(v.at(1), 0.0f);
    EXPECT_FLOAT_EQ(v.at(2), 0.0f);
}

