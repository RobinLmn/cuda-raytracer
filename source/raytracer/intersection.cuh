 #pragma once

#include <glm/glm.hpp>

#include "raytracer/ray.hpp"
#include "raytracer/hit_info.hpp"
#include "raytracer/sphere.hpp"

namespace rAI
{
    __device__ hit_info ray_sphere_intersection(const ray& r, const sphere& s)
    {
        const glm::vec3 oc = r.origin - s.center;
        const float a = glm::dot(r.direction, r.direction);
        const float b = 2.0f * glm::dot(oc, r.direction);
        const float c = glm::dot(oc, oc) - s.radius * s.radius;

        const float discriminant = b * b - 4.0f * a * c;

        if (discriminant < 0.0f)
            return hit_info{ false, 0.0f, glm::vec3{ 0.0f }, glm::vec3{ 0.0f } };

        float distance = (-b - sqrt(discriminant)) / (2.0f * a);

        if (distance < 0.0f)
            return hit_info{ false, 0.0f, glm::vec3{ 0.0f }, glm::vec3{ 0.0f } };

        const glm::vec3 point = r.origin + distance * r.direction;
        const glm::vec3 normal = glm::normalize(point - s.center);
        
        return hit_info{ true, distance, point, normal, s.material };
    }
}
