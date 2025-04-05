#pragma once

#include <glm/vec2.hpp>

#include <cuda_runtime.h>

#include <cmath>

namespace rAI
{
    struct vec2
    {
        __host__ __device__ vec2() : e{ 0, 0 } {}
        __host__ __device__ vec2(float e0, float e1) : e{ e0, e1 } {}
        __host__ __device__ vec2(const vec2& v) : e{ v.e[0], v.e[1] } {}
        __host__ vec2(const glm::vec2& v) : e{ v.x, v.y } {}

        __host__ __device__ inline float x() const { return e[0]; }
        __host__ __device__ inline float y() const { return e[1]; }
        __host__ __device__ inline float u() const { return e[0]; }
        __host__ __device__ inline float v() const { return e[1]; }

        __host__ __device__ inline const vec2& operator+() const { return *this; }
        __host__ __device__ inline vec2 operator-() const { return vec2(-e[0], -e[1]); }            

        __host__ __device__ inline float length() const { return sqrt(e[0] * e[0] + e[1] * e[1]); }
        __host__ __device__ inline float length_squared() const { return e[0] * e[0] + e[1] * e[1]; }
        
        __host__ __device__ inline void normalize() 
        { 
            const float len = length();
            if (len > 0) e[0] /= len; e[1] /= len;
        }

        __host__ __device__ inline static vec2 unit() { return vec2(1.0f, 1.0f); }
        __host__ __device__ inline static vec2 zero() { return vec2(0.0f, 0.0f); }

        __host__ __device__ inline vec2 operator+(const vec2& v) const { return vec2(e[0] + v.e[0], e[1] + v.e[1]); }
        __host__ __device__ inline vec2 operator-(const vec2& v) const { return vec2(e[0] - v.e[0], e[1] - v.e[1]); }
        __host__ __device__ inline vec2 operator/(const vec2& v) const { return vec2(e[0] / v.e[0], e[1] / v.e[1]); }

        __host__ __device__ inline vec2 operator+=(const vec2& v) { e[0] += v.e[0]; e[1] += v.e[1]; return *this; }
        __host__ __device__ inline vec2 operator-=(const vec2& v) { e[0] -= v.e[0]; e[1] -= v.e[1]; return *this; }
        __host__ __device__ inline vec2 operator*=(const vec2& v) { e[0] *= v.e[0]; e[1] *= v.e[1]; return *this; }
        __host__ __device__ inline vec2 operator/=(const vec2& v) { e[0] /= v.e[0]; e[1] /= v.e[1]; return *this; }
        
    private:
        float e[2];
    };

    __host__ __device__ inline float dot(const vec2& v1, const vec2& v2) { return v1.x() * v2.x() + v1.y() * v2.y(); }
    __host__ __device__ inline vec2 mul(const vec2& v1, const vec2& v2) { return vec2(v1.x() * v2.x(), v1.y() * v2.y()); }
}
