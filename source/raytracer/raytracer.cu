#include "raytracer.hpp"

namespace rAI
{
    __global__ void write_to_texture(cudaSurfaceObject_t surface, int width, int height)
    {
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y >= height)
            return;

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= width)
            return;
            
        surf2Dwrite(make_uchar4(255, 0, 0, 255), surface, x * sizeof(uchar4), y);
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
    }
    
    raytracer::~raytracer()
    {
        cudaDestroySurfaceObject(cuda_surface_write);
        cudaFreeArray(cuda_array);
        cudaGraphicsUnregisterResource(cuda_texture_resource);
    }

    void raytracer::render()
    {
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuda_array;
        
        cudaCreateSurfaceObject(&cuda_surface_write, &resDesc);
        
        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
        
        write_to_texture<<<gridDim, blockDim>>>(cuda_surface_write, width, height);
    }

    unsigned int raytracer::get_render_texture() const
    {
        return render_texture.get_id();
    }
}
