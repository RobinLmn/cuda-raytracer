#pragma once

#include <cuda_runtime.h>

namespace rAI
{
    class cuda_texture
    {
    public:
        cuda_texture(const int width, const int height, cudaChannelFormatDesc format);
        ~cuda_texture();

    public:
        cudaSurfaceObject_t get_surface() const;

    private:
        cudaSurfaceObject_t surface;
        cudaArray_t array;
    };
}
