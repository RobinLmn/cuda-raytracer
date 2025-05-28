#pragma once

#include "raytracer/material.hpp"
#include <glm/glm.hpp>
#include <vector>

namespace rAI
{
    struct triangle
    {
        glm::vec3 vertices[3];
        glm::vec3 normals[3];
    };

    struct mesh
    {
        std::vector<triangle> triangles;
        material material;
    };

    struct mesh_info
    {
        int triangle_start;
        int triangle_count;
        material material;
    };
}
