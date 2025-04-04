#include "application.hpp"

#include "core/log.hpp"
#include "core/window.hpp"
#include "core/editor.hpp"

#include "raytracer/raytracer.hpp"

#include "app/viewport.hpp"

namespace app
{
    void application::run()
    {
        core::logger::initialize();

        core::window window{ 1920, 1080, "Engine" };
        core::editor editor;
        rAI::raytracer raytracer{ 800, 800 };

        editor.add_widget<viewport>(raytracer.get_render_texture());

        while (window.is_open())
        {
            window.update();
            raytracer.render();
            editor.render();
        }
    }
}

