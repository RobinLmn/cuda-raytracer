#pragma once

#include "raytracer/texture.hpp"

#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cuda_surface_types.h>

namespace rAI
{
    class raytracer
    {
    public:
        raytracer(const int width, const int height);
        ~raytracer();

    public:
        void render();
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