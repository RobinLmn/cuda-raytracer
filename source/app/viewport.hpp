#pragma once

#include "core/widget.hpp"

#include <chrono>

namespace app
{
    class viewport : public core::widget
    {
    public:
        viewport(const unsigned int render_texture_id, const int width, const int height);

    public:
        void draw() override;

    private:
        const unsigned int render_texture_id;
        const int texture_width;
        const int texture_height;

        std::chrono::high_resolution_clock clock;
        std::chrono::time_point<std::chrono::high_resolution_clock> last_time;
    };
}
