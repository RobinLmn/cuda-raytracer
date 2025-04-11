#include "raytracer/scene.hpp"

#include "core/log.hpp"

namespace rAI
{
    __host__ void upload_scene(scene& scene, const std::vector<sphere>& host_spheres)
    {
        if (!scene.spheres || scene.spheres_count != host_spheres.size())
        {
            if (scene.spheres) 
                cudaFree(scene.spheres);

            scene.spheres_count = host_spheres.size();
            cudaMalloc(&scene.spheres, sizeof(sphere) * scene.spheres_count);
        }

        cudaMemcpy(scene.spheres, host_spheres.data(), sizeof(sphere) * host_spheres.size(), cudaMemcpyHostToDevice);

#ifdef DEBUG
        [[maybe_unused]] cudaError_t error = cudaGetLastError();
        ASSERT(!error, "[CUDA] {} : {}", cudaGetErrorName(error), cudaGetErrorString(error));
#endif
    }
}
