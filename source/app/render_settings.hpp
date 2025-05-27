#pragma once

#include "core/widget.hpp"

#include "raytracer/sky_box.hpp"

namespace app
{
    struct render_settings_data
    {
        int max_bounces;
        int rays_per_pixel;
        
        rAI::sky_box sky_box;
    };

    class render_settings : public core::widget
    {
    public:
        render_settings(render_settings_data& settings);

    public:
        void draw() override;

    private:
        render_settings_data& settings;
    };
}
