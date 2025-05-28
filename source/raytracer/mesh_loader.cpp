#include "mesh_loader.hpp"

#include "core/log.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tinyobjloader/tiny_obj_loader.h>

#include <string>

namespace rAI
{
    std::vector<mesh> load_meshes_from_obj(const char* filename, const material& material)
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename))
        {
            LOG_ERROR("Failed to load OBJ file: {}", filename);
            return {};
        }

        std::vector<mesh> meshes;
        meshes.reserve(shapes.size());

        for (const tinyobj::shape_t& shape : shapes)
        {
            mesh current_mesh;
            current_mesh.material = material;
            current_mesh.triangles.reserve(shape.mesh.num_face_vertices.size());

            size_t index_offset = 0;
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
            {
                const size_t fv = shape.mesh.num_face_vertices[f];
                if (fv != 3)
                {
                    LOG_WARN("Skipping non-triangle face in mesh");
                    index_offset += fv;
                    continue;
                }

                triangle tri;
                for (size_t v = 0; v < 3; v++)
                {
                    const tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

                    tri.vertices[v] = glm::vec3
                    {
                        attrib.vertices[3 * idx.vertex_index + 0],
                        attrib.vertices[3 * idx.vertex_index + 1],
                        attrib.vertices[3 * idx.vertex_index + 2]
                    };

                    if (idx.normal_index >= 0)
                    {
                        tri.normals[v] = glm::vec3
                        {
                            attrib.normals[3 * idx.normal_index + 0],
                            attrib.normals[3 * idx.normal_index + 1],
                            attrib.normals[3 * idx.normal_index + 2]
                        };
                    }
                    else
                    {
                        const glm::vec3& v0 = tri.vertices[0];
                        const glm::vec3& v1 = tri.vertices[1];
                        const glm::vec3& v2 = tri.vertices[2];
                        const glm::vec3& normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                        tri.normals[v] = normal;
                    }
                }

                current_mesh.triangles.push_back(tri);
                index_offset += fv;
            }

            meshes.push_back(std::move(current_mesh));
        }

        return meshes;
    }
}
