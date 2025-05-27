#include "cuda_texture.hpp"

namespace rAI
{
    cuda_texture::cuda_texture(const int width, const int height, cudaChannelFormatDesc format)
    {
        cudaChannelFormatDesc channelDesc = format;
        cudaMallocArray(&array, &channelDesc, width, height, cudaArraySurfaceLoadStore);

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = array;

        cudaCreateSurfaceObject(&surface, &resDesc);
    }

    cuda_texture::~cuda_texture()
    {
        if (surface)
            cudaDestroySurfaceObject(surface);
            
        if (array)
            cudaFreeArray(array);
    }

    cudaSurfaceObject_t cuda_texture::get_surface() const
    {
        return surface;
    }
}
