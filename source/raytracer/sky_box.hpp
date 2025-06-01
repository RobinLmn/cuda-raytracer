#pragma once

#include <glm/glm.hpp>

namespace rAI
{
    struct sky_box
    {
        bool is_hidden;
        
        glm::vec3 ground_color;
        glm::vec3 zenith_color;
        glm::vec3 horizon_color;

        float sun_focus;
        float sun_intensity;
        glm::vec3 sun_direction;
        float brightness;
    };
}
