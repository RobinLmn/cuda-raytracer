#include "raytracer.hpp"

#include "core/log.hpp"

#include "raytracer/ray.hpp"
#include "raytracer/hit_info.hpp"

namespace rAI
{
    __device__ hit_info ray_sphere_intersection(const ray& r, const sphere& s)
    {
        const glm::vec3 oc = r.origin - s.center;
        const float a = glm::dot(r.direction, r.direction);
        const float b = 2.0f * glm::dot(oc, r.direction);
        const float c = glm::dot(oc, oc) - s.radius * s.radius;

        const float discriminant = b * b - 4.0f * a * c;
        if (discriminant < 0.0f)
            return hit_info{ false, 0.0f, glm::vec3(0.0f), glm::vec3(0.0f) };

        float distance = (-b - sqrt(discriminant)) / (2.0f * a);
        if (distance < 0.0f)
            return hit_info{ false, 0.0f, glm::vec3(0.0f), glm::vec3(0.0f) };

        const glm::vec3 point = r.origin + distance * r.direction;
        const glm::vec3 normal = glm::normalize(point - s.center);
        
        return hit_info{ true, distance, point, normal };
    }

    __device__ glm::vec3 trace(const scene& scene, const ray& ray)
    {
        for (int i = 0; i < scene.spheres_count; i++)
        {
            hit_info hit = ray_sphere_intersection(ray, scene.spheres[i]);

            if (hit.did_hit)
                return hit.normal;
        }

        const float a = 0.5f * (ray.direction.y + 1.0f);
        return (1.0f - a) * glm::vec3{ 1.0f, 1.0f, 1.0f } + a * glm::vec3{ 0.5f, 0.7f, 1.0f };
    }

    __global__ void write_to_texture(cudaSurfaceObject_t surface, int width, int height, const rendering_context rendering_context, const scene scene)
    {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (y >= height || x >= width)
            return;

        const glm::vec2 uv = glm::vec2{ (float)x / (float)width, 1.0f - (float)y / (float)height } * 2.0f - 1.0f;
        const glm::vec4 target = rendering_context.inverse_projection_matrix * glm::vec4{ uv, 1.0f, 1.0f };

        const glm::vec3 ray_origin = rendering_context.camera_position;
        const glm::vec3 ray_direction = glm::vec3{ rendering_context.inverse_view_matrix * glm::vec4{ glm::normalize(glm::vec3{ target } / target.w), 0.0f } };

        ray ray{ ray_origin, ray_direction };
        
        const glm::vec3 color = trace(scene, ray);

        uchar4 color_u = make_uchar4(color.r * 255, color.g * 255, color.b * 255, 255);
        surf2Dwrite(color_u, surface, x * sizeof(uchar4), y);
    }

    raytracer::raytracer(const int width, const int height)
        : width{ width }
        , height{ height }
        , cuda_texture_resource{ nullptr }
        , cuda_array{ nullptr }
        , cuda_surface_write{ 0 }
        , render_texture{ width, height }
    {
        cudaGraphicsGLRegisterImage(&cuda_texture_resource, render_texture.get_id(), GL_TEXTURE_2D,  cudaGraphicsRegisterFlagsSurfaceLoadStore);
        cudaGraphicsMapResources(1, &cuda_texture_resource, 0);
        cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_texture_resource, 0, 0);
        cudaGraphicsUnmapResources(1, &cuda_texture_resource, 0);

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuda_array;
        cudaCreateSurfaceObject(&cuda_surface_write, &resDesc);
    }
    
    raytracer::~raytracer()
    {
        cudaDestroySurfaceObject(cuda_surface_write);
        cudaFreeArray(cuda_array);
        cudaGraphicsUnregisterResource(cuda_texture_resource);
    }

    void raytracer::render(const rendering_context& rendering_context, const scene& scene)
    {
        const int thread_x = 16;
        const int thread_y = 16;

        dim3 blocks((width + thread_x - 1) / thread_x, (height + thread_y - 1) / thread_y);
        dim3 threads(thread_x, thread_y);

        write_to_texture<<<blocks, threads>>>(cuda_surface_write, width, height, rendering_context, scene);

#ifdef DEBUG
        [[maybe_unused]] cudaError_t error = cudaGetLastError();
        ASSERT(!error, "[CUDA] {} : {}", cudaGetErrorName(error), cudaGetErrorString(error));
#endif
            
        cudaDeviceSynchronize();
    }

    unsigned int raytracer::get_render_texture() const
    {
        return render_texture.get_id();
    }
}
