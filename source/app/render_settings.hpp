#pragma once

#include "core/widget.hpp"
#include "raytracer/sky_box.hpp"

#include <functional>

namespace app
{
    struct render_settings_data
    {
        bool is_rendering;
        int max_bounces;
        int rays_per_pixel;
        float diverge_strength;
        float defocus_strength;
        float focus_distance;
        
        rAI::sky_box sky_box;
    };

    class render_settings : public core::widget
    {
    public:
        render_settings(render_settings_data& settings, const std::function<void()>& on_save_image_delegate);

    public:
        void draw() override;

    private:
        render_settings_data& settings;
        const std::function<void()> on_save_image_delegate;
    };
}
