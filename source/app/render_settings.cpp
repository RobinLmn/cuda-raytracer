#include "render_settings.hpp"

#include <imgui.h>

#include <glm/gtc/type_ptr.hpp>

namespace app
{
    render_settings::render_settings(render_settings_data& settings)
        : settings(settings)
    {
    }

    void render_settings::draw()
    {
        if (ImGui::Begin("Render Settings"))
        {
            ImGui::Text("Ray Tracing Settings");
            ImGui::Separator();

            ImGui::DragInt("Max Bounces", &settings.max_bounces, 1.0f, 1, 32);
            ImGui::DragInt("Rays per Pixel", &settings.rays_per_pixel, 1.0f, 1, 1000);

            ImGui::Separator();
            ImGui::Text("Sky Box Settings");

            ImGui::ColorEdit3("Ground Color", (float*)glm::value_ptr(settings.sky_box.ground_color));
            ImGui::ColorEdit3("Horizon Color", (float*)glm::value_ptr(settings.sky_box.horizon_color));
            ImGui::ColorEdit3("Zenith Color", (float*)glm::value_ptr(settings.sky_box.zenith_color));
            ImGui::DragFloat("Sun Focus", &settings.sky_box.sun_focus, 1.0f, 0.f, 1024.f);
            ImGui::DragFloat("SunIntensity", &settings.sky_box.sun_intensity, 1.0f, 0.f, 500.f);
        }
        ImGui::End();
    }
}
