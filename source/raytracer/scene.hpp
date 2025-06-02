#pragma once

#include "raytracer/sphere.hpp"
#include "raytracer/mesh.hpp"

#include <glm/glm.hpp>
#include <vector>

namespace rAI
{
    struct scene
    {
        sphere* spheres = nullptr;
        int spheres_count = 0;

        triangle* triangles = nullptr;
        int triangles_count = 0;

        mesh_info* meshes_info = nullptr;
        int meshes_count = 0;
    };

    void upload_scene(scene& scene, const std::vector<sphere>& spheres, const std::vector<mesh>& meshes);
    void free_scene(scene& scene);
}
