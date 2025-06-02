#pragma once

#include "raytracer/texture.hpp"
#include "raytracer/scene.hpp"
#include "raytracer/sky_box.hpp"
#include "raytracer/cuda_texture.hpp"

#include <glm/glm.hpp>

namespace rAI
{
    struct rendering_context
    {
        glm::vec3 camera_position;
        glm::mat4 inverse_view_matrix;
        glm::mat4 inverse_projection_matrix;

        int max_bounces;
        int rays_per_pixel;
        float diverge_strength;
        float defocus_strength;
        float focus_distance;

        sky_box sky_box;
        
        float exposure;
        float gamma;
    };

    class raytracer
    {
    public:
        raytracer(const int width, const int height);

    public:
        void render(const rendering_context& rendering_context, const scene& scene, const bool should_accumulate);
        void reset_accumulation();
        
        void resize(const int new_width, const int new_height);
        const texture& get_render_texture() const;

    private:
        texture render_texture;
        cuda_texture accumulation_texture;

        int width;
        int height;

        int frame_index;
    };
}