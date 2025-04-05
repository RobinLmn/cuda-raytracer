#pragma once

#include "vec4.hpp"

#include <cuda_runtime.h>
#include <glm/mat4x4.hpp>

namespace rAI
{
    struct mat4
    {
    public: 
        __host__ __device__ inline mat4() : data{ 0.0f } {}
        __host__ __device__ inline mat4(const glm::mat4& m) : data{ m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3], m[2][0], m[2][1], m[2][2], m[2][3], m[3][0], m[3][1], m[3][2], m[3][3] } {}

        __host__ __device__ inline mat4(const float arr[16])
        {
            for (int i = 0; i < 16; i++) data[i] = arr[i];
        }

    public:
        __host__ __device__ inline float& at(int row, int col)
        {
            return data[col * 4 + row];
        }

        __host__ __device__ inline float at(int row, int col) const
        {
            return data[col * 4 + row];
        }

    private:
        float data[16];
    };

    __host__ __device__ inline mat4 mul(const mat4& a, const mat4& b)
    {
        mat4 result;
        
        for (int row = 0; row < 4; ++row)
        {
            for (int col = 0; col < 4; ++col)
            {
                float sum = 0.0f;
                for (int i = 0; i < 4; i++)
                    sum += a.at(row, i) * b.at(i, col);

                result.at(row, col) = sum;
            }
        }        
        
        return result;
    }

    __host__ __device__ inline vec4 mul(const mat4& m, const vec4& v)
    {
        return vec4(
            m.at(0, 0) * v.x() + m.at(0, 1) * v.y() + m.at(0, 2) * v.z() + m.at(0, 3) * v.w(),
            m.at(1, 0) * v.x() + m.at(1, 1) * v.y() + m.at(1, 2) * v.z() + m.at(1, 3) * v.w(),
            m.at(2, 0) * v.x() + m.at(2, 1) * v.y() + m.at(2, 2) * v.z() + m.at(2, 3) * v.w(),
            m.at(3, 0) * v.x() + m.at(3, 1) * v.y() + m.at(3, 2) * v.z() + m.at(3, 3) * v.w()
        );
    }
} 