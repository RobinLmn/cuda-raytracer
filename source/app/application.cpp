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

        std::vector<rAI::sphere> spheres;
        spheres.push_back(rAI::sphere{ glm::vec3{ 1.0f, 1.0f, 0.0f }, 0.6f });
        spheres.push_back(rAI::sphere{ glm::vec3{ -1.0f, -1.0f, 0.0f }, 0.3f });

        rAI::scene scene;
        rAI::upload_scene(scene, spheres);

        app::camera camera{ 0.1f, 100.0f, 45.f, 1085.f / 1026.f };
        
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

            rAI::rendering_context rendering_context{ camera.get_position(), camera.get_inverse_view_matrix(), camera.get_inverse_projection_matrix() };
            
            raytracer.render(rendering_context, scene);
            
            editor.render();
        }
    }
}

