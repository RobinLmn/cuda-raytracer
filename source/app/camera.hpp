#pragma once

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

namespace app
{
    class camera
    {
    public:
        camera(const float near_plane, const float far_plane, const float fov, const float aspect);

    public:
        void update(float delta_time);

        glm::vec3 get_position() const;
        glm::mat4 get_inverse_view_matrix() const;
        glm::mat4 get_inverse_projection_matrix() const;
        
        void set_speed(const float speed);
        void set_sensitivity(const float sensitivity);

    private:
        void update_inverse_view_matrix();
        void update_inverse_projection_matrix();

    private:
        float near_plane;
        float far_plane;

        float fov;
        float aspect;

        glm::vec3 position;
        glm::vec3 direction;

        constexpr static glm::vec3 up{ 0.0f, 1.0f, 0.0f };

        float speed;
        float sensitivity;
        
        glm::vec2 last_mouse_position;

        glm::mat4 inverse_view_matrix;
        glm::mat4 inverse_projection_matrix;
    };
}

