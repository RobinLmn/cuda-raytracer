#pragma once

#include <glm/glm.hpp>
#include <curand_kernel.h>

namespace rAI
{
    __device__ float random_float(curandState& random_state)
    {
        return curand_uniform(&random_state);
    }

    __device__ float random_normal(curandState& random_state)
    {
        return curand_normal(&random_state);
    }

    __device__ glm::vec3 random_direction(curandState& random_state)
    {
        const float x = random_normal(random_state);
        const float y = random_normal(random_state);
        const float z = random_normal(random_state);
        return glm::normalize(glm::vec3{ x, y, z });
    }

    __device__ glm::vec3 random_hemisphere_direction(const glm::vec3& normal, curandState& random_state)
    {
        const glm::vec3 direction = random_direction(random_state);
        const float sign = glm::dot(direction, normal) > 0.0f ? 1.0f : -1.0f;
        return direction * sign;
    }

    __device__ glm::vec2 random_point_in_circle(curandState& random_state)
    {
        float angle = random_float(random_state) * 2 * 3.1415926f;
        glm::vec2 point_on_circle{ cos(angle), sin(angle) };
        return point_on_circle * sqrt(random_float(random_state));
    }
}
