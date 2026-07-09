#include <gtest/gtest.h>

#include "math.hpp"


TEST(VectorTest, DefaultConstructor)
{
    Vector v;

    EXPECT_EQ(v.size(), 0);
}


TEST(VectorTest, Constructor)
{
    Vector v(10);

    EXPECT_EQ(v.size(), 10);
}


TEST(VectorTest, ElementAccess)
{
    Vector v(3);

    v[0] = 1.0f;
    v[1] = 2.0f;
    v[2] = 3.0f;

    EXPECT_FLOAT_EQ(v[0], 1.0f);
    EXPECT_FLOAT_EQ(v[1], 2.0f);
    EXPECT_FLOAT_EQ(v[2], 3.0f);
}


TEST(VectorTest, ConstElementAccess)
{
    const Vector v(3);

    EXPECT_NO_THROW(v[0]);
}


TEST(VectorTest, Addition)
{
    Vector a(3);
    Vector b(3);

    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;

    b[0] = 4.0f;
    b[1] = 5.0f;
    b[2] = 6.0f;


    Vector c = a + b;


    EXPECT_FLOAT_EQ(c[0], 5.0f);
    EXPECT_FLOAT_EQ(c[1], 7.0f);
    EXPECT_FLOAT_EQ(c[2], 9.0f);
}


TEST(VectorTest, Subtraction)
{
    Vector a(3);
    Vector b(3);

    a[0] = 5.0f;
    a[1] = 7.0f;
    a[2] = 9.0f;

    b[0] = 1.0f;
    b[1] = 2.0f;
    b[2] = 3.0f;


    Vector c = a - b;


    EXPECT_FLOAT_EQ(c[0], 4.0f);
    EXPECT_FLOAT_EQ(c[1], 5.0f);
    EXPECT_FLOAT_EQ(c[2], 6.0f);
}


TEST(VectorTest, ScalarMultiplication)
{
    Vector v(3);

    v[0] = 1.0f;
    v[1] = 2.0f;
    v[2] = 3.0f;


    Vector result = v * 2.0f;


    EXPECT_FLOAT_EQ(result[0], 2.0f);
    EXPECT_FLOAT_EQ(result[1], 4.0f);
    EXPECT_FLOAT_EQ(result[2], 6.0f);
}


TEST(VectorTest, ScalarMultiplicationReverse)
{
    Vector v(3);

    v[0] = 1.0f;
    v[1] = 2.0f;
    v[2] = 3.0f;


    Vector result = 2.0f * v;


    EXPECT_FLOAT_EQ(result[0], 2.0f);
    EXPECT_FLOAT_EQ(result[1], 4.0f);
    EXPECT_FLOAT_EQ(result[2], 6.0f);
}


TEST(VectorTest, AdditionAssignment)
{
    Vector a(3);
    Vector b(3);

    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;

    b[0] = 4.0f;
    b[1] = 5.0f;
    b[2] = 6.0f;


    a += b;


    EXPECT_FLOAT_EQ(a[0], 5.0f);
    EXPECT_FLOAT_EQ(a[1], 7.0f);
    EXPECT_FLOAT_EQ(a[2], 9.0f);
}


TEST(VectorTest, SubtractionAssignment)
{
    Vector a(3);
    Vector b(3);

    a[0] = 5.0f;
    a[1] = 7.0f;
    a[2] = 9.0f;

    b[0] = 1.0f;
    b[1] = 2.0f;
    b[2] = 3.0f;


    a -= b;


    EXPECT_FLOAT_EQ(a[0], 4.0f);
    EXPECT_FLOAT_EQ(a[1], 5.0f);
    EXPECT_FLOAT_EQ(a[2], 6.0f);
}


TEST(VectorTest, ScalarMultiplicationAssignment)
{
    Vector v(3);

    v[0] = 1.0f;
    v[1] = 2.0f;
    v[2] = 3.0f;


    v *= 2.0f;


    EXPECT_FLOAT_EQ(v[0], 2.0f);
    EXPECT_FLOAT_EQ(v[1], 4.0f);
    EXPECT_FLOAT_EQ(v[2], 6.0f);
}


TEST(VectorTest, DotProduct)
{
    Vector a(3);
    Vector b(3);

    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;

    b[0] = 4.0f;
    b[1] = 5.0f;
    b[2] = 6.0f;


    EXPECT_FLOAT_EQ(a.dot(b), 32.0f);
}


TEST(VectorTest, Sum)
{
    Vector v(3);

    v[0] = 1.0f;
    v[1] = 2.0f;
    v[2] = 3.0f;


    EXPECT_FLOAT_EQ(v.sum(), 6.0f);
}


TEST(VectorTest, Mean)
{
    Vector v(4);

    v[0] = 2.0f;
    v[1] = 4.0f;
    v[2] = 6.0f;
    v[3] = 8.0f;


    EXPECT_FLOAT_EQ(v.mean(), 5.0f);
}


TEST(VectorTest, Norm)
{
    Vector v(3);

    v[0] = 3.0f;
    v[1] = 4.0f;
    v[2] = 0.0f;


    EXPECT_FLOAT_EQ(v.norm(), 5.0f);
}


TEST(VectorTest, Zero)
{
    Vector v(3);

    v[0] = 1.0f;
    v[1] = 2.0f;
    v[2] = 3.0f;


    v.zero();


    EXPECT_FLOAT_EQ(v[0], 0.0f);
    EXPECT_FLOAT_EQ(v[1], 0.0f);
    EXPECT_FLOAT_EQ(v[2], 0.0f);
}


TEST(VectorTest, AdditionDifferentSizesThrows)
{
    Vector a(3);
    Vector b(4);


    EXPECT_THROW(
        a + b,
        std::invalid_argument
    );
}


TEST(VectorTest, SubtractionDifferentSizesThrows)
{
    Vector a(3);
    Vector b(4);


    EXPECT_THROW(
        a - b,
        std::invalid_argument
    );
}


TEST(VectorTest, DotDifferentSizesThrows)
{
    Vector a(3);
    Vector b(4);


    EXPECT_THROW(
        a.dot(b),
        std::invalid_argument
    );
}


TEST(VectorTest, OutOfRangeAccessThrows)
{
    Vector v(3);

    EXPECT_THROW(
        v[10],
        std::out_of_range
    );
}

