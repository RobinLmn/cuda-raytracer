#include "camera.hpp"

#include "core/log.hpp"
#include "core/input.hpp"

#include <glfw/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>

namespace app
{
    camera::camera(const float near_plane, const float fov, const float aspect)
        : near_plane(near_plane)
        , fov(fov)
        , aspect(aspect)
        , position(0.0f, 0.0f, 5.0f)
        , direction(0.0f, 0.0f, -1.0f)
        , yaw(180.0f)
        , pitch(0.0f)
        , speed(10.0f)
        , sensitivity(0.1f)
    {
    }

    void camera::update(float delta_time)
    {
        glm::vec2 mouse_pos = core::input_manager::get_mouse_position();

        glm::vec3 right = glm::cross(up, direction);

        if (core::input_manager::is_key_pressed(GLFW_KEY_D))
            position += right * speed * delta_time;
        if (core::input_manager::is_key_pressed(GLFW_KEY_A))
            position -= right * speed * delta_time;
        if (core::input_manager::is_key_pressed(GLFW_KEY_W))
            position += direction * speed * delta_time;
        if (core::input_manager::is_key_pressed(GLFW_KEY_S))
            position -= direction * speed * delta_time;
        if (core::input_manager::is_key_pressed(GLFW_KEY_SPACE))
            position += up * speed * delta_time;
        if (core::input_manager::is_key_pressed(GLFW_KEY_LEFT_SHIFT))
            position -= up * speed * delta_time;

		if (core::input_manager::is_mouse_button_pressed(GLFW_MOUSE_BUTTON_2))
		{
			increment_yaw((mouse_pos.x - last_mouse_position.x) * sensitivity);
			increment_pitch((last_mouse_position.y - mouse_pos.y) * sensitivity);
            recalculate_direction();
		}

        last_mouse_position = mouse_pos;
    }

    void camera::increment_yaw(float delta)
    {
        yaw += delta;
    }

    void camera::increment_pitch(float delta)
    {
        pitch += delta;
        pitch = std::clamp(pitch, -89.0f, 89.0f);
    }

    void camera::recalculate_direction()
    {
        const float yaw_rad = glm::radians(yaw);
        const float pitch_rad = glm::radians(pitch);

        direction = glm::vec3{ sin(yaw_rad) * cos(pitch_rad), sin(pitch_rad), cos(yaw_rad) * cos(pitch_rad) };
        direction = glm::normalize(direction);
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

    glm::mat4 camera::get_local_to_world() const
    {
        glm::vec3 right = glm::normalize(glm::cross(up, direction));

        glm::mat4 model = glm::mat4(1.0f);
        model[0] = glm::vec4(-right, 0.0f);
        model[1] = glm::vec4(up, 0.0f);
        model[2] = glm::vec4(-direction, 0.0f);
        model[3] = glm::vec4(position, 1.0f);

        return model;
    }

    glm::vec3 camera::get_view_params() const
    {
        float viewplane_height = 2.0f * near_plane * tan(glm::radians(fov * 0.5f));
        float viewplane_width = viewplane_height * aspect;

        return glm::vec3{ viewplane_width, viewplane_height, near_plane };
    }
}
