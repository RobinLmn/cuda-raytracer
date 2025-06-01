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
    camera::camera(const float near_plane, const float far_plane, const float fov, const float aspect, const glm::vec3& position, const glm::vec3& direction, const float speed, const float sensitivity)
        : near_plane{ near_plane }
        , far_plane{ far_plane }
        , fov{ fov }
        , aspect{ aspect }
        , position{ position }
        , direction{ direction }
        , speed{ speed }
        , sensitivity{ sensitivity }
    {
        update_inverse_projection_matrix();
        update_inverse_view_matrix();
    }

    void camera::update(const float delta_time)
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
            const glm::vec2 delta = mouse_pos - last_mouse_position;

            const float pitch_delta = delta.y * sensitivity;
            const float yaw_delta = delta.x * sensitivity;
            const glm::quat rotation = glm::normalize(glm::cross(glm::angleAxis(-pitch_delta, right), glm::angleAxis(-yaw_delta, up)));

            direction = glm::rotate(rotation, direction);

            is_dirty = true;
		}

        last_mouse_position = mouse_pos;

        if (is_dirty)
            update_inverse_view_matrix();
    }

    void camera::update_inverse_view_matrix()
    {
        inverse_view_matrix = glm::inverse(glm::lookAt(position, position + direction, up));
    }

    void camera::update_inverse_projection_matrix()
    {
        inverse_projection_matrix = glm::inverse(glm::perspective(glm::radians(fov), aspect, near_plane, far_plane));
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
