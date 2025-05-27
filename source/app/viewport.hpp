#pragma once

#include "core/widget.hpp"

#include <chrono>

namespace app
{
    class viewport : public core::widget
    {
    public:
        viewport(const unsigned int render_texture_id);

    public:
        void draw() override;

    private:
        unsigned int render_texture_id;

        std::chrono::high_resolution_clock clock;
        std::chrono::time_point<std::chrono::high_resolution_clock> last_time;
    };
}
