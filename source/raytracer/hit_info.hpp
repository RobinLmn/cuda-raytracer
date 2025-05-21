#pragma once

#include "raytracer/material.hpp"

#include <glm/glm.hpp>

namespace rAI
{
    struct hit_info
    {
        bool did_hit;
        float distance;
        glm::vec3 point;
        glm::vec3 normal;
        material material;
    };
}
