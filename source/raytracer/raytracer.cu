#include "raytracer.hpp"

#include "math/ray.hpp"
#include "math/vec4.hpp"

namespace rAI
{
    __global__ void write_to_texture(cudaSurfaceObject_t surface, int width, int height, const rendering_context rendering_context)
    {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (y >= height || x >= width)
            return;
        
        const vec2 uv = vec2{ float(x) / width - 0.5f, float(y) / height - 0.5f };

        vec3 view_point_local = mul(vec3{ uv, 1.0f }, rendering_context.camera_view);
        vec4 view_point = mul(rendering_context.camera_local_to_world, vec4{ view_point_local, 1.0f });

        ray r{ rendering_context.camera_position, view_point.xyz() - rendering_context.camera_position};
        r.direction.normalize();

        vec3 color = r.direction;
        
        uchar4 color_u = make_uchar4(color.r() * 255, color.g() * 255, color.b() * 255, 255);
        surf2Dwrite(color_u, surface, x * sizeof(uchar4), y);
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

    void raytracer::render(const rendering_context& rendering_context)
    {
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuda_array;
        
        cudaCreateSurfaceObject(&cuda_surface_write, &resDesc);
        
        const int thread_x = 16;
        const int thread_y = 16;

        dim3 blocks((width + thread_x - 1) / thread_x, (height + thread_y - 1) / thread_y);
        dim3 threads(thread_x, thread_y);

        write_to_texture<<<blocks, threads>>>(cuda_surface_write, width, height, rendering_context);

        cudaDeviceSynchronize();
    }

    unsigned int raytracer::get_render_texture() const
    {
        return render_texture.get_id();
    }
}
