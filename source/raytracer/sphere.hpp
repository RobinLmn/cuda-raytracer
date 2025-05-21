#pragma once

#include "raytracer/material.hpp"

#include <glm/glm.hpp>

namespace rAI
{
    struct sphere
    {
        glm::vec3 center;
        float radius;
        material material;
    };
}
