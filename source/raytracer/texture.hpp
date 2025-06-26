#pragma once

#include <glad/glad.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cuda_surface_types.h>

#include <vector>

namespace rAI
{
    class texture
    {
    public:
        texture(const int width, const int height);
        ~texture();

    public:
        void map();
        void unmap();

        unsigned int get_id() const;

        int get_width() const;
        int get_height() const;

        cudaSurfaceObject_t get_surface() const;
        std::vector<unsigned char> read_pixels() const;

    private:
        const int width;
        const int height;

        unsigned int id;
        unsigned int unit;

        cudaGraphicsResource* cuda_texture_resource;
        cudaSurfaceObject_t cuda_surface_write;
        cudaArray_t cuda_array;
    };
}
