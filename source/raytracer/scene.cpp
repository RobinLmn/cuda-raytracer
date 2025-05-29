#include "raytracer/scene.hpp"

#include <cuda_runtime.h>

namespace
{
    using namespace rAI;

    void upload_spheres(scene& scene, const std::vector<sphere>& spheres)
    {
        scene.spheres_count = static_cast<int>(spheres.size());
        if (scene.spheres_count == 0)
            return;

        cudaMalloc(&scene.spheres, scene.spheres_count * sizeof(sphere));
        cudaMemcpy(scene.spheres, spheres.data(), scene.spheres_count * sizeof(sphere), cudaMemcpyHostToDevice);
    }

    void upload_meshes(scene& scene, const std::vector<mesh>& meshes)
    {
        scene.meshes_count = static_cast<int>(meshes.size());
        if (scene.meshes_count == 0)
            return;

        for (const mesh& mesh : meshes)
            scene.triangles_count += mesh.triangles.size();

        std::vector<triangle> triangles(scene.triangles_count);
        std::vector<mesh_info> meshes_info(scene.meshes_count);

        int triangle_start = 0;
        int mesh_index = 0;
        for (const mesh& mesh : meshes)
        {
            mesh_info mesh_info{ triangle_start, static_cast<int>(mesh.triangles.size()), mesh.material, mesh.bounding_box };

            std::copy(mesh.triangles.begin(), mesh.triangles.end(), triangles.begin() + triangle_start);
            meshes_info[mesh_index] = mesh_info;

            triangle_start += static_cast<int>(mesh.triangles.size());
            mesh_index++;
        }

        cudaMalloc(&scene.triangles, scene.triangles_count * sizeof(triangle));
        cudaMemcpy(scene.triangles, triangles.data(), scene.triangles_count * sizeof(triangle), cudaMemcpyHostToDevice);

        cudaMalloc(&scene.meshes_info, scene.meshes_count * sizeof(mesh_info));
        cudaMemcpy(scene.meshes_info, meshes_info.data(), scene.meshes_count * sizeof(mesh_info), cudaMemcpyHostToDevice);
    }
}

namespace rAI
{
    void upload_scene(scene& scene, const std::vector<sphere>& spheres, const std::vector<mesh>& meshes)
    {
        free_scene(scene);
        
        upload_spheres(scene, spheres);
        upload_meshes(scene, meshes);
    }

    void free_scene(scene& scene)
    {
        if (scene.spheres)
        {
            cudaFree(scene.spheres);
            scene.spheres = nullptr;
        }
        
        if (scene.triangles)
        {
            cudaFree(scene.triangles);
            scene.triangles = nullptr;
        }

        if (scene.meshes_info)
        {
            cudaFree(scene.meshes_info);
            scene.meshes_info = nullptr;
        }
        
        scene.spheres_count = 0;
        scene.triangles_count = 0;
        scene.meshes_count = 0;
    }
}
