#pragma once

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

namespace app
{
    class camera
    {
    public:
        camera(const float near_plane, const float far_plane, const float fov, const float aspect, const glm::vec3& position, const glm::vec3& direction, const float speed, const float sensitivity);

    public:
        void update(const float delta_time);

        glm::vec3 get_position() const;
        glm::mat4 get_inverse_view_matrix() const;
        glm::mat4 get_inverse_projection_matrix() const;

    private:
        void update_inverse_view_matrix();
        void update_inverse_projection_matrix();

    private:
        static constexpr glm::vec3 up{ 0.f, 1.f, 0.f };

        float near_plane;
        float far_plane;

        float fov;
        float aspect;

        glm::vec3 position;
        glm::vec3 direction;

        float speed;
        float sensitivity;
        
        glm::vec2 last_mouse_position;

        glm::mat4 inverse_view_matrix;
        glm::mat4 inverse_projection_matrix;
    };
}

