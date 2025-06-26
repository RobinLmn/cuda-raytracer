#include "raytracer.hpp"

#include "core/log.hpp"

#include "raytracer/random.cuh"
#include "raytracer/intersection.cuh"
#include "raytracer/cuda_utils.cuh"

#include <curand_kernel.h>

namespace rAI
{
    __device__ glm::vec3 get_sky_light(const sky_box& sky, const ray& ray)
    {
        if (sky.is_hidden)
            return glm::vec3{ 0.f };
            
        const float horizon_to_zenith_gradient_t = powf(glm::smoothstep(0.0f, 0.4f, ray.direction.y), 0.35f);
        const glm::vec3 horizon_to_zenith_gradient = glm::mix(sky.horizon_color, sky.zenith_color, horizon_to_zenith_gradient_t);

        const float ground_to_sky_gradient_t = glm::smoothstep(-0.01f, 0.0f, ray.direction.y);
        const glm::vec3 ground_to_sky_gradient = glm::mix(sky.ground_color, horizon_to_zenith_gradient, ground_to_sky_gradient_t);

        const float sun = powf(max(0.0f, glm::dot(ray.direction, glm::normalize(sky.sun_direction))), sky.sun_focus) * sky.sun_intensity;
        
        return (ground_to_sky_gradient + sun) * sky.brightness;
    }

    __device__ glm::vec3 aces_tone_map(const glm::vec3& color)
    {
        const float a = 2.51f;
        const float b = 0.03f;
        const float c = 2.43f;
        const float d = 0.59f;
        const float e = 0.14f;

        return glm::clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0f, 1.0f);
    }

    __device__ void post_process(glm::vec3& color, const rendering_context rendering_context)
    {
        color *= rendering_context.exposure;
        color = aces_tone_map(color);
        color = glm::pow(color, glm::vec3{ 1.0f / rendering_context.gamma });
    }

    __device__ hit_info get_closest_hit(const scene& scene, const ray& ray)
    {
        hit_info closest_hit{ false, FLT_MAX, glm::vec3{ 0.f }, glm::vec3{ 0.f } };

        for (int sphere_index = 0; sphere_index < scene.spheres_count; sphere_index++)
        {
            const sphere& sphere = scene.spheres[sphere_index];
            
            const hit_info& hit = ray_sphere_intersection(ray, sphere);
            if (hit.did_hit && hit.distance < closest_hit.distance)
            {
                closest_hit = hit;
                closest_hit.material = sphere.material;
            }
        }

        for (int mesh_index = 0; mesh_index < scene.meshes_count; mesh_index++)
        {
            const mesh_info& mesh_info = scene.meshes_info[mesh_index];

            const hit_info& bounding_box_hit = ray_aabb_intersection(ray, mesh_info.bounding_box);
            if (!bounding_box_hit.did_hit || bounding_box_hit.distance >= closest_hit.distance)
                continue;

            for (int triangle_index = 0; triangle_index < mesh_info.triangle_count; triangle_index++)
            {
                const triangle& triangle = scene.triangles[mesh_info.triangle_start + triangle_index];
                const hit_info& hit = ray_triangle_intersection(ray, triangle);

                if (hit.did_hit && hit.distance < closest_hit.distance)
                {
                    closest_hit = hit;
                    closest_hit.material = mesh_info.material;
                }
            }
        }

        return closest_hit;
    }

    __device__ glm::vec3 trace(const scene& scene, const ray& starting_ray, uint32_t& random_state, const int max_bounces, const sky_box& sky_box)
    {
        glm::vec3 incoming_light{ 0.f };
        glm::vec3 ray_color{ 1.f };
        ray ray = starting_ray;

        for (int bounce = 0; bounce < max_bounces; bounce++)
        {
            const hit_info& closest_hit = get_closest_hit(scene, ray);
            if (closest_hit.did_hit)
            {
                const glm::vec3 diffuse = glm::normalize(closest_hit.normal + random_direction(random_state));
                const glm::vec3 specular = glm::reflect(ray.direction, closest_hit.normal);

                const bool specular_bounce = closest_hit.material.specular_probability >= random_float(random_state);

                ray.origin = closest_hit.point;
                ray.direction = glm::normalize(glm::mix(diffuse, specular, closest_hit.material.smoothness * specular_bounce));

                incoming_light += closest_hit.material.emission_strength * closest_hit.material.emission_color * ray_color;
                ray_color *= specular_bounce ? closest_hit.material.specular_color : closest_hit.material.color;

                float p = max(ray_color.r, max(ray_color.g, ray_color.b));
                if (p < 0.001f || random_float(random_state) > p)
                    break;

                ray_color /= p; 
            }
            else
            {
                incoming_light += get_sky_light(sky_box, ray) * ray_color;
                break;
            }
        }

        return incoming_light;
    }

    __global__ void write_to_texture(cudaSurfaceObject_t output_surface, glm::vec3* accumulation_texture, int width, int height, const rendering_context rendering_context, const scene scene, const int frame_index, const bool should_accumulate)
    {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int x = blockIdx.x * blockDim.x + threadIdx.x;

        if (y >= height || x >= width)
            return;

        const glm::vec2 uv = glm::vec2{ static_cast<float>(x) / static_cast<float>(width), 1.0f - static_cast<float>(y) / static_cast<float>(height) } * 2.f - 1.f;
        const glm::vec4 target = rendering_context.inverse_projection_matrix * glm::vec4{ uv, 1.0f, 1.0f };

        const int pixel_index = y + width * x;
        uint32_t random_state = pixel_index + frame_index * 719393;
        
        glm::vec3 incoming_light = glm::vec3{ 0.f };

        const glm::vec3 direction = glm::vec3{ rendering_context.inverse_view_matrix * glm::vec4{ glm::normalize(glm::vec3{ target } / target.w), 0.f } };
        const glm::vec3 right = glm::normalize(glm::cross(direction, glm::vec3{ 0.f, 1.f, 0.f }));
        const glm::vec3 up = glm::normalize(glm::cross(right, direction));

        const glm::vec3 focal_point = rendering_context.camera_position + direction * rendering_context.focus_distance;

        for (int ray_index = 0; ray_index < rendering_context.rays_per_pixel; ray_index++)
        {
            const glm::vec2 jitter = random_point_in_circle(random_state) * rendering_context.diverge_strength / static_cast<float>(width);
            const glm::vec3 jittered_focal_point = focal_point + right * jitter.x + up * jitter.y;
            const glm::vec2 defocus_jitter = random_point_in_circle(random_state) * rendering_context.defocus_strength / static_cast<float>(width);

            const glm::vec3 ray_origin = rendering_context.camera_position + right * defocus_jitter.x + up * defocus_jitter.y;
            const glm::vec3 direction = jittered_focal_point - ray_origin;
            
            // Prevent division by zero in normalization
            if (glm::length(direction) < 1e-6f)
                continue;
                
            const glm::vec3 ray_direction = glm::normalize(direction);

            ray ray{ ray_origin, ray_direction };
            incoming_light += trace(scene, ray, random_state, rendering_context.max_bounces, rendering_context.sky_box);
        }

        incoming_light /= static_cast<float>(rendering_context.rays_per_pixel);

        if (!should_accumulate)
        {
            post_process(incoming_light, rendering_context);

            const uchar4 new_color_u = make_uchar4(incoming_light.r * 255, incoming_light.g * 255, incoming_light.b * 255, 255);
            surf2Dwrite(new_color_u, output_surface, x * sizeof(uchar4), y);

            return;
        }

    	accumulation_texture[pixel_index] = (accumulation_texture[pixel_index] * static_cast<float>(frame_index) + incoming_light) / static_cast<float>(frame_index + 1);

        glm::vec3 average_color = accumulation_texture[pixel_index];
        post_process(average_color, rendering_context);
        
        const uchar4 average_color_u = make_uchar4(average_color.x * 255, average_color.y * 255, average_color.z * 255, 255);
        surf2Dwrite(average_color_u, output_surface, x * sizeof(uchar4), y);
    }

    __host__ raytracer::raytracer(const int width, const int height)
        : width{ width }
        , height{ height }
        , render_texture{ width, height }
        , accumulation_texture{}
        , frame_index{ 0 }
    {
        const size_t render_size = width * height * sizeof(glm::vec3);
        cudaMalloc(&accumulation_texture, render_size);
        cudaMemset(accumulation_texture, 0, render_size);

        CUDA_VALIDATE();
    }

    __host__ void raytracer::reset_accumulation()
    {
        const size_t render_size = width * height * sizeof(glm::vec3);
        cudaMemset(accumulation_texture, 0, render_size);

        CUDA_VALIDATE();
        frame_index = 0;
    }

    __host__ void raytracer::render(const rendering_context& rendering_context, const scene& scene, const bool should_accumulate)
    {
        const int thread_x = 16;
        const int thread_y = 16;

        dim3 blocks((width + thread_x - 1) / thread_x, (height + thread_y - 1) / thread_y);
        dim3 threads(thread_x, thread_y);

        render_texture.map();
        write_to_texture<<<blocks, threads>>>(render_texture.get_surface(), accumulation_texture, width, height, rendering_context, scene, frame_index, should_accumulate);
        render_texture.unmap();

        CUDA_VALIDATE();

        if (should_accumulate)
            frame_index++;
    }

    __host__ const texture& raytracer::get_render_texture() const
    {
        return render_texture;
    }
}
