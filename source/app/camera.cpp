#include "camera.hpp"

#include "core/log.hpp"
#include "core/input.hpp"

#include <glfw/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

namespace app
{
    camera::camera(const float near_plane, const float fov, const float aspect)
        : near_plane(near_plane)
        , fov(fov)
        , aspect(aspect)
        , position(0.0f, 0.0f, 0.0f)
        , direction(1.0f, 0.0f, 0.0f)
        , yaw(0.0f)
        , pitch(0.0f)
        , speed(10.0f)
        , sensitivity(0.1f)
        , up(0.0f, 1.0f, 0.0f)
    {
    }

    void camera::update(float delta_time)
    {
        glm::vec2 mouse_pos = core::input_manager::get_mouse_position();

        if (core::input_manager::is_key_pressed(GLFW_KEY_D))
            position.x -= speed * delta_time;
        if (core::input_manager::is_key_pressed(GLFW_KEY_A))
            position.x += speed * delta_time;
        if (core::input_manager::is_key_pressed(GLFW_KEY_W))
            position.z += speed * delta_time;
        if (core::input_manager::is_key_pressed(GLFW_KEY_S))
            position.z -= speed * delta_time;
        if (core::input_manager::is_key_pressed(GLFW_KEY_LEFT_SHIFT))
            position.y += speed * delta_time;
        if (core::input_manager::is_key_pressed(GLFW_KEY_SPACE))
            position.y -= speed * delta_time;

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
    }

    void camera::recalculate_direction()
    {
        direction = glm::vec3{ cos(glm::radians(yaw)) * cos(glm::radians(pitch)), sin(glm::radians(pitch)), sin(glm::radians(yaw)) * cos(glm::radians(pitch)) };
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
        return glm::lookAt(position, position + direction, up);
    }

    glm::vec3 camera::get_view() const
    {
        float viewplane_height = 2.0f * near_plane * tan(glm::radians(fov * 0.5f));
        float viewplane_width = viewplane_height * aspect;

        return glm::vec3{ viewplane_width, viewplane_height, near_plane };
    }
}
