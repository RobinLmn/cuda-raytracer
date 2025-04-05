#include "application.hpp"

#include "core/log.hpp"
#include "core/window.hpp"
#include "core/editor.hpp"

#include "raytracer/raytracer.hpp"

#include "app/camera.hpp"
#include "app/viewport.hpp"

#include <chrono>

namespace app
{
    void application::run()
    {
        core::window window{ 1920, 1080, "Engine" };
        core::editor editor;
        rAI::raytracer raytracer{ 1085, 1026 };

        app::camera camera{ 0.f, 45.f, 1085.f / 1026.f };
        
        editor.add_widget<viewport>(raytracer.get_render_texture());

        const auto clock = std::chrono::high_resolution_clock{};
        auto last_time = clock.now();

        while (window.is_open())
        {
            using seconds = std::chrono::duration<float, std::ratio<1>>;
        
            float delta_time = std::chrono::duration_cast<seconds>(clock.now() - last_time).count();
            last_time = clock.now();

            window.update();

            camera.update(delta_time);

            rAI::rendering_context rendering_context{ camera.get_position(), camera.get_view(), camera.get_local_to_world() };
            raytracer.render(rendering_context);
            
            editor.render();
        }
    }
}

