#include "render_settings.hpp"

#include <imgui.h>

#include <glm/gtc/type_ptr.hpp>

#include <glad/glad.h>

namespace app
{
    render_settings::render_settings(render_settings_data& settings, const std::function<void()>& on_save_image_delegate)
        : settings{ settings }
        , on_save_image_delegate{ on_save_image_delegate }
    {
    }

    void render_settings::draw()
    {
        if (ImGui::Begin("Render Settings"))
        {
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 4));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 4));
            
            const float window_width = ImGui::GetWindowWidth();
            const float button_width = 60.0f;
            const float total_width = button_width * 2 + ImGui::GetStyle().ItemSpacing.x;
            ImGui::SetCursorPosX((window_width - total_width) * 0.5f);
            
            if (ImGui::Button(settings.is_rendering ? "Stop" : "Render", ImVec2(button_width, 24)))
            {
                settings.is_rendering = !settings.is_rendering;
            }

            ImGui::SameLine();

            if (ImGui::Button("Save", ImVec2(button_width, 24)))
                on_save_image_delegate();
            
            ImGui::PopStyleVar(2);
            ImGui::Separator();

            ImGui::Text("Ray Tracing Settings");
            ImGui::Separator();

            ImGui::DragInt("Max Bounces", &settings.max_bounces, 1.0f, 1, 32);
            ImGui::DragInt("Rays per Pixel", &settings.rays_per_pixel, 1.0f, 1, 1000);

            ImGui::Separator();
            ImGui::Text("Sky Box Settings");

            ImGui::Checkbox("Is Hidden", &settings.sky_box.is_hidden);
            ImGui::ColorEdit3("Ground Color", (float*)glm::value_ptr(settings.sky_box.ground_color));
            ImGui::ColorEdit3("Horizon Color", (float*)glm::value_ptr(settings.sky_box.horizon_color));
            ImGui::ColorEdit3("Zenith Color", (float*)glm::value_ptr(settings.sky_box.zenith_color));
            ImGui::DragFloat("Sun Focus", &settings.sky_box.sun_focus, 1.0f, 0.f, 1024.f);
            ImGui::DragFloat("SunIntensity", &settings.sky_box.sun_intensity, 1.0f, 0.f, 500.f);
        }
        ImGui::End();
    }
}
