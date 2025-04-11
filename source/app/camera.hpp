#pragma once

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

namespace app
{
    class camera
    {
    public:
        camera(const float near_plane, const float fov, const float aspect);

    public:
        void update(float delta_time);

        glm::vec3 get_position() const;
        glm::vec3 get_view_params() const;
        glm::mat4 get_local_to_world() const;
        
        void set_speed(const float speed);
        void set_sensitivity(const float sensitivity);

    private:
        void increment_yaw(float delta);
        void increment_pitch(float delta);

        void recalculate_direction();

    private:
        float near_plane;
        float fov;
        float aspect;

        glm::vec3 position;
        glm::vec3 direction;
        const glm::vec3 up{ 0.0f, 1.0f, 0.0f };

    private:
        float yaw;
        float pitch;

    private:
        float speed;
        float sensitivity;

    private:
        glm::vec2 last_mouse_position;
    };
}

