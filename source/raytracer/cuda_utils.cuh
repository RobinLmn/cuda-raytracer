#pragma once

__device__ inline float4 operator+(const float4& a, const float4& b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ inline float4 operator*(const float4& a, const float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

__device__ inline float4 operator/(const float4& a, const float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
