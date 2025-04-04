#pragma once

#include "core/widget.hpp"

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
    };
}
