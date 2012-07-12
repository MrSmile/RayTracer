// vec3d.h : 3d vector mathematics
//

#pragma once

#include <cmath>



template<typename real> struct Vec3D
{
    real x, y, z;


    Vec3D()
    {
    }

    Vec3D(const Vec3D &v) : x(v.x), y(v.y), z(v.z)
    {
    }

    Vec3D(real x_, real y_, real z_) : x(x_), y(y_), z(z_)
    {
    }

    Vec3D &operator = (const Vec3D &v)
    {
        x = v.x;  y = v.y;  z = v.z;  return *this;
    }

    Vec3D operator + (const Vec3D &v) const
    {
        return Vec3D(x + v.x, y + v.y, z + v.z);
    }

    Vec3D &operator += (const Vec3D &v)
    {
        x += v.x;  y += v.y;  z += v.z;  return *this;
    }

    Vec3D operator - (const Vec3D &v) const
    {
        return Vec3D(x - v.x, y - v.y, z - v.z);
    }

    Vec3D &operator -= (const Vec3D &v)
    {
        x -= v.x;  y -= v.y;  z -= v.z;  return *this;
    }

    Vec3D operator - () const
    {
        return Vec3D(-x, -y, -z);
    }

    Vec3D operator * (real a) const
    {
        return Vec3D(a * x, a * y, a * z);
    }

    Vec3D &operator *= (real a)
    {
        x *= a;  y *= a;  z *= a;  return *this;
    }

    real operator * (const Vec3D &v) const
    {
        return x * v.x + y * v.y + z * v.z;
    }

    Vec3D operator / (real a) const
    {
        return (*this) * (1 / a);
    }

    Vec3D operator /= (real a)
    {
        return (*this) *= (1 / a);
    }

    Vec3D operator ^ (const Vec3D &v) const
    {
        return Vec3D(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }

    Vec3D &operator ^= (const Vec3D &v)
    {
        *this = *this ^ v;  return *this;
    }

    real sqr() const
    {
        return x * x + y * y + z * z;
    }

    real len() const
    {
        return std::sqrt(x * x + y * y + z * z);
    }
};

template<typename real> inline Vec3D<real> operator * (real a, const Vec3D<real> &v)
{
    return v * a;
}
