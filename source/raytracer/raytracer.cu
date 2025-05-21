#include "raytracer.hpp"

#include "core/log.hpp"

#include "raytracer/ray.hpp"
#include "raytracer/hit_info.hpp"

namespace rAI
{
    __device__ float random_pcg_float(int& random_state)
    {
        random_state = random_state * 747796405 + 2891336453;
        unsigned int result = ((random_state >> ((random_state >> 28) + 4)) ^ random_state) * 277803737;
        result = (result >> 22) ^ result;
        return (float)(result) / 4294967295.0f;
    }

    __device__ float random_normal(int& random_state)
    {
        const float theta = 2.0f * 3.1415926f * random_pcg_float(random_state);
        const float rho = sqrt(-2.0f * log(random_pcg_float(random_state)));
        return rho * cos(theta);
    }

    __device__ glm::vec3 random_direction(int& random_state)
    {
        const float x = random_normal(random_state);
        const float y = random_normal(random_state);
        const float z = random_normal(random_state);
        return glm::normalize(glm::vec3{ x, y, z });
    }

    __device__ glm::vec3 random_hemisphere_direction(const glm::vec3& normal, int& random_state)
    {
        const glm::vec3 direction = random_direction(random_state);
        const float sign = glm::dot(direction, normal) > 0.0f ? 1.0f : -1.0f;
        return direction * sign;
    }

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
        
        return hit_info{ true, distance, point, normal, s.material };
    }

    __device__ hit_info get_closest_hit(const scene& scene, const ray& ray)
    {
        hit_info closest_hit = hit_info{ false, FLT_MAX, glm::vec3(0.0f), glm::vec3(0.0f) };
        for (int i = 0; i < scene.spheres_count; i++)
        {
            hit_info hit = ray_sphere_intersection(ray, scene.spheres[i]);

            if (hit.did_hit && hit.distance < closest_hit.distance)
                closest_hit = hit;
        }

        return closest_hit;
    }

    __device__ glm::vec3 trace(const scene& scene, const ray& starting_ray, int& random_state)
    {
        const int max_bounces = 20;

        glm::vec3 incoming_light = glm::vec3{ 0.0f };
        glm::vec3 ray_color = glm::vec3{ 1.0f };
        ray ray = starting_ray;

        for ( int i = 0; i < max_bounces; i++)
        {
            const hit_info closest_hit = get_closest_hit(scene, ray);
            if (closest_hit.did_hit)
            {
                ray.origin = closest_hit.point;
                ray.direction = random_hemisphere_direction(closest_hit.normal, random_state);

                incoming_light += closest_hit.material.emission_strength * closest_hit.material.emission_color * ray_color;
                ray_color *= closest_hit.material.color;
            }
            else
            {
                break;
            }
        }

        return incoming_light;
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
        
        int random_state = y + width * x;

        glm::vec3 color = glm::vec3{ 0.0f };
        const int rays_per_pixel = 100;

        for (int i = 0; i < rays_per_pixel; i++)
        {
            color += trace(scene, ray, random_state);
        }

        color /= rays_per_pixel;

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
