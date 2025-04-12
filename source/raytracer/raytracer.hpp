#pragma once

#include "raytracer/texture.hpp"
#include "raytracer/scene.hpp"

#include <glad/glad.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cuda_surface_types.h>

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
        ~raytracer();

    public:
        void render(const rendering_context& rendering_context, const scene& scene);
        
        void resize(const int new_width, const int new_height);
        unsigned int get_render_texture() const;

    private:
        texture render_texture;

        int width;
        int height;
        
        cudaGraphicsResource_t cuda_texture_resource;
        cudaArray* cuda_array;
        cudaSurfaceObject_t cuda_surface_write;
    };
}