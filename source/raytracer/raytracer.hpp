#pragma once

#include "raytracer/texture.hpp"
#include "raytracer/scene.hpp"

#include <glm/glm.hpp>  

namespace rAI
{
    struct rendering_context
    {
        glm::vec3 camera_position;
        glm::mat4 inverse_view_matrix;
        glm::mat4 inverse_projection_matrix;
    };

    class raytracer
    {
    public:
        raytracer(const int width, const int height);

    public:
        void render(const rendering_context& rendering_context, const scene& scene);
        
        void resize(const int new_width, const int new_height);
        unsigned int get_render_texture() const;

    private:
        texture render_texture;

        int width;
        int height;
    };
}