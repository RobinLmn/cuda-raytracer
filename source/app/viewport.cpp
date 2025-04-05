#include "viewport.hpp"

#include <imgui.h>

namespace app
{
    viewport::viewport(const unsigned int render_texture_id)
        : render_texture_id{ render_texture_id }
    {
    }

    void viewport::draw()
    {
        ImGui::Begin("Viewport");
        {
            ImGui::Image((void*)(intptr_t)render_texture_id, ImVec2(1085, 1026));
        }
        ImGui::End();
    }
}
