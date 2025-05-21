#pragma once

#include <glm/glm.hpp>

namespace rAI
{
    __device__ float random_pcg_float(int& random_state)
    {
        random_state = random_state * 747796405 + 2891336453;
        unsigned int result = ((random_state >> ((random_state >> 28) + 4)) ^ random_state) * 277803737;
        result = (result >> 22) ^ result;
        return (float)(result) / 4294967295.0f;
    }

    __device__ float random_normal(int& random_state)
    {
        const float theta = 2.0f * 3.1415926f * random_pcg_float(random_state);
        const float rho = sqrt(-2.0f * log(random_pcg_float(random_state)));
        return rho * cos(theta);
    }

    __device__ glm::vec3 random_direction(int& random_state)
    {
        const float x = random_normal(random_state);
        const float y = random_normal(random_state);
        const float z = random_normal(random_state);
        return glm::normalize(glm::vec3{ x, y, z });
    }

    __device__ glm::vec3 random_hemisphere_direction(const glm::vec3& normal, int& random_state)
    {
        const glm::vec3 direction = random_direction(random_state);
        const float sign = glm::dot(direction, normal) > 0.0f ? 1.0f : -1.0f;
        return direction * sign;
    }
}
