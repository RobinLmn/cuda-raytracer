#pragma once

#include "raytracer/sphere.hpp"
#include "raytracer/mesh.hpp"

#include <glm/glm.hpp>
#include <vector>

namespace rAI
{
    struct scene
    {
        sphere* spheres;
        int spheres_count;

        triangle* triangles;
        int triangles_count;

        mesh_info* meshes_info;
        int meshes_count;
    };

    void upload_scene(scene& scene, const std::vector<sphere>& spheres, const std::vector<mesh>& meshes);
    void free_scene(scene& scene);
}
