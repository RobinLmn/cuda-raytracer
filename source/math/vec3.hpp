#pragma once

#include "vec2.hpp"
#include <glm/vec3.hpp>

#include <cuda_runtime.h>

#include <cmath>

namespace rAI
{
    struct vec3
    {
        __host__ __device__ vec3() : e{ 0, 0, 0 } {}
        __host__ __device__ vec3(float e0, float e1, float e2) : e{ e0, e1, e2 } {}
        __host__ __device__ vec3(const vec2& v, float z) : e{ v.x(), v.y(), z } {}
        __host__ __device__ vec3(const vec3& v) : e{ v.e[0], v.e[1], v.e[2] } {}
        __host__ vec3(const glm::vec3& v) : e{ v.x, v.y, v.z } {}

        __host__ __device__ inline float x() const { return e[0]; }
        __host__ __device__ inline float y() const { return e[1]; }
        __host__ __device__ inline float z() const { return e[2]; }
        __host__ __device__ inline float r() const { return e[0]; }
        __host__ __device__ inline float g() const { return e[1]; }
        __host__ __device__ inline float b() const { return e[2]; }

        __host__ __device__ inline const vec3& operator+() const { return *this; }
        __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
        
        __host__ __device__ inline float length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
        __host__ __device__ inline float length_squared() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }

        __host__ __device__ inline void normalize() 
        { 
            const float len = length();
            if (len > 0) e[0] /= len; e[1] /= len; e[2] /= len;
        }

        __host__ __device__ inline static vec3 unit() { return vec3(1.0f, 1.0f, 1.0f); }
        __host__ __device__ inline static vec3 zero() { return vec3(0.0f, 0.0f, 0.0f); }

        __host__ __device__ inline vec3 operator+(const vec3& v) const { return vec3(e[0] + v.e[0], e[1] + v.e[1], e[2] + v.e[2]); }
        __host__ __device__ inline vec3 operator-(const vec3& v) const { return vec3(e[0] - v.e[0], e[1] - v.e[1], e[2] - v.e[2]); }
        __host__ __device__ inline vec3 operator/(const vec3& v) const { return vec3(e[0] / v.e[0], e[1] / v.e[1], e[2] / v.e[2]); }

        __host__ __device__ inline vec3 operator+=(const vec3& v) { e[0] += v.e[0]; e[1] += v.e[1]; e[2] += v.e[2]; return *this; }
        __host__ __device__ inline vec3 operator-=(const vec3& v) { e[0] -= v.e[0]; e[1] -= v.e[1]; e[2] -= v.e[2]; return *this; }
        __host__ __device__ inline vec3 operator*=(const vec3& v) { e[0] *= v.e[0]; e[1] *= v.e[1]; e[2] *= v.e[2]; return *this; }
        __host__ __device__ inline vec3 operator/=(const vec3& v) { e[0] /= v.e[0]; e[1] /= v.e[1]; e[2] /= v.e[2]; return *this; }
        
        __host__ __device__ inline vec2 xy() const { return vec2(e[0], e[1]); }
        
    private:
        float e[3];
    };

    __host__ __device__ inline float dot(const vec3& v1, const vec3& v2) { return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z(); }
    __host__ __device__ inline vec3 cross(const vec3& v1, const vec3& v2) { return vec3(v1.y() * v2.z() - v1.z() * v2.y(), v1.z() * v2.x() - v1.x() * v2.z(), v1.x() * v2.y() - v1.y() * v2.x()); }
    __host__ __device__ inline vec3 mul(const vec3& v1, const vec3& v2) { return vec3(v1.x() * v2.x(), v1.y() * v2.y(), v1.z() * v2.z()); }

}
