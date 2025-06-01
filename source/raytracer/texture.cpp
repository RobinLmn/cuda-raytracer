#include "texture.hpp"

#include "core/log.hpp"

#include <vector>

namespace rAI
{
    texture::texture(const int width, const int height)
        : width{ width }
        , height{ height }
        , id{ 0 }
        , unit{ 0 }
        , cuda_surface_write{ 0 }
    {
        glActiveTexture(GL_TEXTURE0 + unit);

        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        cudaGraphicsResource_t cuda_texture_resource;

        cudaGraphicsGLRegisterImage(&cuda_texture_resource, id, GL_TEXTURE_2D,  cudaGraphicsRegisterFlagsSurfaceLoadStore);
        cudaGraphicsMapResources(1, &cuda_texture_resource, 0);
        cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_texture_resource, 0, 0);
        cudaGraphicsUnmapResources(1, &cuda_texture_resource, 0);

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuda_array;
        cudaCreateSurfaceObject(&cuda_surface_write, &resDesc);

        CUDA_VALIDATE();
    }

    texture::~texture()
    {
        glDeleteTextures(1, &id);

        if (cuda_surface_write)
            cudaDestroySurfaceObject(cuda_surface_write);

        if (cuda_array)
            cudaFreeArray(cuda_array);
    }
    
    void texture::bind() const
    {
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, id);
    }

    void texture::unbind() const
    {
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    unsigned int texture::get_id() const
    {
        return id;
    }

    unsigned int texture::get_unit() const
    {
        return unit;
    }

    cudaSurfaceObject_t texture::get_surface() const
    {
        return cuda_surface_write;
    }

    std::vector<unsigned char> texture::read_pixels() const
    {
        glFinish();
        
        glBindTexture(GL_TEXTURE_2D, id);

        std::vector<unsigned char> pixels(width * height * 4);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

        glBindTexture(GL_TEXTURE_2D, 0);

        return pixels;
    }
}
