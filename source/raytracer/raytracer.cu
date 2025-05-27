#include "raytracer.hpp"

#include "core/log.hpp"

#include "raytracer/random.cuh"
#include "raytracer/intersection.cuh"

#include <curand_kernel.h>

namespace rAI
{
    __device__ glm::vec3 get_sky_light(const sky_box& sky, const ray& ray)
    {
        const float horizon_to_zenith_gradient_t = powf(glm::smoothstep(0.0f, 0.4f, ray.direction.y), 0.35f);
        const glm::vec3 horizon_to_zenith_gradient = glm::mix(sky.horizon_color, sky.zenith_color, horizon_to_zenith_gradient_t);

        const float ground_to_sky_gradient_t = glm::smoothstep(-0.01f, 0.0f, ray.direction.y);
        const glm::vec3 ground_to_sky_gradient = glm::mix(sky.ground_color, horizon_to_zenith_gradient, ground_to_sky_gradient_t);

        const float sun = powf(max(0.0f, glm::dot(ray.direction, glm::normalize(sky.sun_direction))), sky.sun_focus) * sky.sun_intensity;
        
        return ground_to_sky_gradient + sun;
    }

    __device__ hit_info get_closest_hit(const scene& scene, const ray& ray)
    {
        hit_info closest_hit{ false, FLT_MAX, glm::vec3{ 0.0f }, glm::vec3{ 0.0f } };

        for (int i = 0; i < scene.spheres_count; i++)
        {
            hit_info hit = ray_sphere_intersection(ray, scene.spheres[i]);

            if (hit.did_hit && hit.distance < closest_hit.distance)
                closest_hit = hit;
        }

        return closest_hit;
    }

    __device__ glm::vec3 trace(const scene& scene, const ray& starting_ray, curandState& random_state, const int max_bounces, const sky_box& sky_box)
    {
        glm::vec3 incoming_light{ 0.0f };
        glm::vec3 ray_color{ 1.0f };
        ray ray = starting_ray;

        for (int i = 0; i < max_bounces; i++)
        {
            const hit_info& closest_hit = get_closest_hit(scene, ray);
            if (closest_hit.did_hit)
            {
                ray.origin = closest_hit.point;
                ray.direction = random_hemisphere_direction(closest_hit.normal, random_state);

                incoming_light += closest_hit.material.emission_strength * closest_hit.material.emission_color * ray_color;
                ray_color *= closest_hit.material.color;
            }
            else
            {
                incoming_light += get_sky_light(sky_box, ray) * ray_color;
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
        
        curandState random_state;
        curand_init(y + width * x, 0, 0, &random_state);
        
        glm::vec3 incoming_light = glm::vec3{ 0.0f };

        for (int i = 0; i < rendering_context.rays_per_pixel; i++)
        {
            const glm::vec3 ray_origin = rendering_context.camera_position;
            const glm::vec3 ray_direction = glm::vec3{ rendering_context.inverse_view_matrix * glm::vec4{ glm::normalize(glm::vec3{ target } / target.w), 0.0f } };

            ray ray{ ray_origin, ray_direction };

            incoming_light += trace(scene, ray, random_state, rendering_context.max_bounces, rendering_context.sky_box);
        }

        incoming_light /= static_cast<float>(rendering_context.rays_per_pixel);

        uchar4 color_u = make_uchar4(incoming_light.r * 255, incoming_light.g * 255, incoming_light.b * 255, 255);
        surf2Dwrite(color_u, surface, x * sizeof(uchar4), y);
    }

    __host__ raytracer::raytracer(const int width, const int height)
        : width{ width }
        , height{ height }
        , render_texture{ width, height }
    {
    }

    __host__ void raytracer::render(const rendering_context& rendering_context, const scene& scene)
    {
        const int thread_x = 16;
        const int thread_y = 16;

        dim3 blocks((width + thread_x - 1) / thread_x, (height + thread_y - 1) / thread_y);
        dim3 threads(thread_x, thread_y);

        write_to_texture<<<blocks, threads>>>(render_texture.get_surface_write(), width, height, rendering_context, scene);

        cudaDeviceSynchronize();
    }

    __host__ unsigned int raytracer::get_render_texture() const
    {
        return render_texture.get_id();
    }
}
