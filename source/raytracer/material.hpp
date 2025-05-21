#pragma once

#include <glm/glm.hpp>

namespace rAI
{
    struct material
    {
        glm::vec3 color;
        
        glm::vec3 emission_color;
        float emission_strength;
    };
}
