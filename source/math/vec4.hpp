#pragma once

#include "vec2.hpp"
#include "vec3.hpp"

#include <glm/vec4.hpp>

#include <cuda_runtime.h>

#include <cmath>

namespace rAI
{
    struct vec4
    {
        __host__ __device__ vec4() : e{ 0, 0, 0, 0 } {}
        __host__ __device__ vec4(float e0, float e1, float e2, float e3) : e{ e0, e1, e2, e3 } {}
        __host__ __device__ vec4(const vec3& v, float w) : e{ v.x(), v.y(), v.z(), w } {}
        __host__ __device__ vec4(const vec2& v, float z, float w) : e{ v.x(), v.y(), z, w } {}
        __host__ __device__ vec4(const vec4& v) : e{ v.e[0], v.e[1], v.e[2], v.e[3] } {}
        __host__ vec4(const glm::vec4& v) : e{ v.x, v.y, v.z, v.w } {}

        __host__ __device__ inline float x() const { return e[0]; }
        __host__ __device__ inline float y() const { return e[1]; }
        __host__ __device__ inline float z() const { return e[2]; }
        __host__ __device__ inline float w() const { return e[3]; }
        __host__ __device__ inline float r() const { return e[0]; }
        __host__ __device__ inline float g() const { return e[1]; }
        __host__ __device__ inline float b() const { return e[2]; }
        __host__ __device__ inline float a() const { return e[3]; }

        __host__ __device__ inline const vec4& operator+() const { return *this; }
        __host__ __device__ inline vec4 operator-() const { return vec4(-e[0], -e[1], -e[2], -e[3]); }

        __host__ __device__ inline float length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3]); }
        __host__ __device__ inline float length_squared() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3]; }

        __host__ __device__ inline void normalize() 
        { 
            const float len = length();
            if (len > 0) e[0] /= len; e[1] /= len; e[2] /= len; e[3] /= len;
        }

        __host__ __device__ inline static vec4 unit() { return vec4(1.0f, 1.0f, 1.0f, 1.0f); }
        __host__ __device__ inline static vec4 zero() { return vec4(0.0f, 0.0f, 0.0f, 0.0f); }

        __host__ __device__ inline vec4 operator+(const vec4& v) const { return vec4(e[0] + v.e[0], e[1] + v.e[1], e[2] + v.e[2], e[3] + v.e[3]); }
        __host__ __device__ inline vec4 operator-(const vec4& v) const { return vec4(e[0] - v.e[0], e[1] - v.e[1], e[2] - v.e[2], e[3] - v.e[3]); }
        __host__ __device__ inline vec4 operator/(const vec4& v) const { return vec4(e[0] / v.e[0], e[1] / v.e[1], e[2] / v.e[2], e[3] / v.e[3]); }

        __host__ __device__ inline vec4 operator+=(const vec4& v) { e[0] += v.e[0]; e[1] += v.e[1]; e[2] += v.e[2]; e[3] += v.e[3]; return *this; }
        __host__ __device__ inline vec4 operator-=(const vec4& v) { e[0] -= v.e[0]; e[1] -= v.e[1]; e[2] -= v.e[2]; e[3] -= v.e[3]; return *this; }
        __host__ __device__ inline vec4 operator*=(const vec4& v) { e[0] *= v.e[0]; e[1] *= v.e[1]; e[2] *= v.e[2]; e[3] *= v.e[3]; return *this; }
        __host__ __device__ inline vec4 operator/=(const vec4& v) { e[0] /= v.e[0]; e[1] /= v.e[1]; e[2] /= v.e[2]; e[3] /= v.e[3]; return *this; }
        
        __host__ __device__ inline vec3 xyz() const { return vec3(e[0], e[1], e[2]); }
        __host__ __device__ inline vec2 xy() const { return vec2(e[0], e[1]); }
        
    private:
        float e[4];
    };

    __host__ __device__ inline float dot(const vec4& v1, const vec4& v2) { return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z() + v1.w() * v2.w(); }
    __host__ __device__ inline vec4 mul(const vec4& v1, const vec4& v2) { return vec4(v1.x() * v2.x(), v1.y() * v2.y(), v1.z() * v2.z(), v1.w() * v2.w()); }
} 