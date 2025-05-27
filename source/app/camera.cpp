#include "camera.hpp"

#include "core/log.hpp"
#include "core/input.hpp"

#include <glfw/glfw3.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include <algorithm>

namespace app
{
    camera::camera(const float near_plane, const float far_plane, const float fov, const float aspect)
        : near_plane{ near_plane }
        , far_plane{ far_plane }
        , fov{ fov }
        , aspect{ aspect }
        , position{ 0.0f, 2.7f, 3.15f }
        , direction{ 0.0f, -0.14f, -1.0f }
        , speed{ 6.0f }
        , sensitivity{ 0.005f }
    {
        update_inverse_projection_matrix();
        update_inverse_view_matrix();
    }

    void camera::update(float delta_time)
    {
        const glm::vec2 mouse_pos = core::input_manager::get_mouse_position();
        const glm::vec3 right = glm::cross(direction, up);

        bool is_dirty = false;

        if (core::input_manager::is_key_pressed(GLFW_KEY_D))
        {
            position += right * speed * delta_time;
            is_dirty = true;
        }
        if (core::input_manager::is_key_pressed(GLFW_KEY_A))
        {
            position -= right * speed * delta_time;
            is_dirty = true;
        }
        if (core::input_manager::is_key_pressed(GLFW_KEY_W))
        {
            position += direction * speed * delta_time;
            is_dirty = true;
        }
        if (core::input_manager::is_key_pressed(GLFW_KEY_S))
        {
            position -= direction * speed * delta_time;
            is_dirty = true;
        }
        if (core::input_manager::is_key_pressed(GLFW_KEY_SPACE))
        {
            position += up * speed * delta_time;
            is_dirty = true;
        }
        if (core::input_manager::is_key_pressed(GLFW_KEY_LEFT_SHIFT))
        {
            position -= up * speed * delta_time;
            is_dirty = true;
        }

		if (core::input_manager::is_mouse_button_pressed(GLFW_MOUSE_BUTTON_2))
		{
            glm::vec2 delta = mouse_pos - last_mouse_position;

            float pitch_delta = delta.y * sensitivity;
            float yaw_delta = delta.x * sensitivity;

            glm::quat rotation = glm::normalize(glm::cross(glm::angleAxis(-pitch_delta, right), glm::angleAxis(-yaw_delta, up)));
            direction = glm::rotate(rotation, direction);

            is_dirty = true;
		}

        last_mouse_position = mouse_pos;

        if (is_dirty)
        {
            update_inverse_view_matrix();
        }
    }

    void camera::update_inverse_view_matrix()
    {
        inverse_view_matrix = glm::inverse(glm::lookAt(position, position + direction, up));
    }

    void camera::update_inverse_projection_matrix()
    {
        inverse_projection_matrix = glm::inverse(glm::perspective(glm::radians(fov), aspect, near_plane, far_plane));
    }

    void camera::set_speed(const float speed)
    {
        this->speed = speed;
    }

    void camera::set_sensitivity(const float sensitivity)
    {
        this->sensitivity = sensitivity;
    }

    glm::vec3 camera::get_position() const
    {
        return position;
    }

    glm::mat4 camera::get_inverse_view_matrix() const
    {
        return inverse_view_matrix;
    }

    glm::mat4 camera::get_inverse_projection_matrix() const
    {
        return inverse_projection_matrix;
    }
}
