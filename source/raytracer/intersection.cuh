#pragma once

#include <glm/glm.hpp>

#include "raytracer/ray.hpp"
#include "raytracer/hit_info.hpp"
#include "raytracer/sphere.hpp"
#include "raytracer/mesh.hpp"
#include "raytracer/aabb.hpp"

namespace rAI
{
    __device__ hit_info ray_sphere_intersection(const ray &r, const sphere &s)
    {
        const glm::vec3 oc = r.origin - s.center;
        const float a = glm::dot(r.direction, r.direction);
        const float b = 2.0f * glm::dot(oc, r.direction);
        const float c = glm::dot(oc, oc) - s.radius * s.radius;

        const float discriminant = b * b - 4.0f * a * c;

        if (discriminant < 0.0f)
            return hit_info{false, 0.0f, glm::vec3{0.0f}, glm::vec3{0.0f}};

        float distance = (-b - sqrt(discriminant)) / (2.0f * a);

        if (distance < 0.0f)
            return hit_info{false, 0.0f, glm::vec3{0.0f}, glm::vec3{0.0f}};

        const glm::vec3 hit_point = r.origin + distance * r.direction;
        const glm::vec3 hit_normal = glm::normalize(hit_point - s.center);

        return hit_info{ true, distance, hit_point, hit_normal };
    }

    __device__ hit_info ray_triangle_intersection(const ray& ray, const triangle& triangle)
    {
        const glm::vec3& A = triangle.vertices[0];
        const glm::vec3& B = triangle.vertices[1];
        const glm::vec3& C = triangle.vertices[2];

        const glm::vec3 AB = B - A;
        const glm::vec3 AC = C - A;

        const glm::vec3 normal = glm::cross(AB, AC);
        const glm::vec3 AO = ray.origin - A;
        const glm::vec3 DAO = glm::cross(AO, ray.direction);

        const float determinant = -glm::dot(ray.direction, normal);
        const float determinant_inverse = 1.f / determinant;

        const float distance = glm::dot(AO, normal) * determinant_inverse;
        const float u = glm::dot(AC, DAO) * determinant_inverse;
        const float v = -glm::dot(AB, DAO) * determinant_inverse;
        const float w = 1 - u - v;

        const bool did_hit = determinant >= 1e-6f && distance >= 0 && u >= 0 && v >= 0 && w >= 0;

        if (!did_hit)
            return hit_info{false, 0.0f, glm::vec3{0.0f}, glm::vec3{0.0f}};

        const glm::vec3 hit_point = ray.origin + distance * ray.direction;
        const glm::vec3 hit_normal = glm::normalize(triangle.normals[0] * w + triangle.normals[1] * u + triangle.normals[2] * v);

        return hit_info{true, distance, hit_point, hit_normal };
    }

    __device__ hit_info ray_aabb_intersection(const ray& ray, const aabb& aabb)
    {
        const glm::vec3 direction_inverse = 1.f / ray.direction;

        const glm::vec3 t_min = (aabb.min - ray.origin) * direction_inverse;
        const glm::vec3 t_max = (aabb.max - ray.origin) * direction_inverse;

        const glm::vec3 t_1 = min(t_min, t_max);
        const glm::vec3 t_2 = max(t_min, t_max);

        const float t_near = max(max(t_1.x, t_1.y), t_1.z);
        const float t_far = min(min(t_2.x, t_2.y), t_2.z);
        
        return hit_info{ t_near <= t_far };
    }
}
