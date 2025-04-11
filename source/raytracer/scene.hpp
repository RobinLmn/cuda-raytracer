#pragma once

#include "raytracer/sphere.hpp"

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <vector>

namespace rAI
{
    struct scene
    {
        sphere* spheres = nullptr;
        size_t spheres_count = 0;
    };

    __host__ void upload_scene(scene& scene, const std::vector<sphere>& spheres);
}
