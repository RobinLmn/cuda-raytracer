#pragma once

#include "raytracer/mesh.hpp"

#include <vector>

namespace rAI
{
    std::vector<mesh> load_meshes_from_obj(const char* filename, const material& material);
}
