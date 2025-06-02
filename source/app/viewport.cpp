#include "viewport.hpp"

#include <imgui.h>

namespace app
{
    viewport::viewport(const unsigned int render_texture_id, const int width, const int height)
        : render_texture_id{ render_texture_id }
        , texture_width{ width }
        , texture_height{ height }
    {
    }

    void viewport::draw()
    {
        using seconds = std::chrono::duration<float, std::ratio<1>>;
        
        const float delta_time = std::chrono::duration_cast<seconds>(clock.now() - last_time).count();
        last_time = clock.now();

        ImGui::Begin("Viewport");
        {
            ImGui::Image((void*)(intptr_t)render_texture_id, ImVec2{ static_cast<float>(texture_width), static_cast<float>(texture_height) });

            ImGui::SetCursorPos(ImVec2(texture_width - 70, 35));
            ImGui::BeginGroup();
            {
                ImGui::Text("FPS: %.1f", 1.f / delta_time);
            }
            ImGui::EndGroup();
        }
        ImGui::End();
    }
}
