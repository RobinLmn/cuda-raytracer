#pragma once

#include <glm/glm.hpp>

namespace rAI
{
    __device__ int next_random(uint32_t& random_state)
    {
        const uint32_t state = random_state * 747796405u + 2891336453u;
        const uint32_t word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    [[nodiscard]] __device__ uint32_t random_int(uint32_t& random_state)
    {
        random_state = 1664525u * random_state + 1013904223u;
        return random_state;
    }

    __device__ float random_float(uint32_t& random_state)
    {
        return static_cast<float>(random_int(random_state) & 0x00FFFFFF) / static_cast<float>(0x01000000);
    }

    __device__ float random_normal(uint32_t& random_state)
    {
        float theta = 2.f * 3.1415926f * random_float(random_state);
        float u = random_float(random_state);
        
        u = max(u, 1e-7f);
        float rho = sqrt(-2 * log(u));
        return rho * cos(theta);
    }

    __device__ glm::vec3 random_direction(uint32_t& random_state)
    {
        const float x = random_normal(random_state);
        const float y = random_normal(random_state);
        const float z = random_normal(random_state);
        return glm::normalize(glm::vec3{ x, y, z });
    }

    __device__ glm::vec2 random_point_in_circle(uint32_t& random_state)
    {
        float angle = random_float(random_state) * 2 * 3.1415926f;
        glm::vec2 point_on_circle{ cos(angle), sin(angle) };
        return point_on_circle * sqrt(random_float(random_state));
    }
}
