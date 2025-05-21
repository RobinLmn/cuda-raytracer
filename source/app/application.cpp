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
        rAI::scene scene;

        {
            std::vector<rAI::sphere> spheres;

            rAI::sphere sphere_ground{ glm::vec3{ 0.0f, -100.5f, -1.0f }, 100.0f, rAI::material{ glm::vec3{ 1.0f }, glm::vec3{ 0.0f }, 0.0f } };

            rAI::sphere sphere_a = rAI::sphere{ glm::vec3{ 0.0f, 0.0f, -1.2f }, 0.5f, rAI::material{ glm::vec3{ 1.0f, 0.0f, 0.0f }, glm::vec3{ 0.0f }, 0.0f } };
            rAI::sphere sphere_b = rAI::sphere{ glm::vec3{ -1.0f, 0.0f, -1.0f }, 0.5f, rAI::material{ glm::vec3{ 0.0f, 1.0f, 0.0f }, glm::vec3{ 0.0f }, 0.0f } };
            rAI::sphere sphere_c = rAI::sphere{ glm::vec3{ 1.0f, 0.0f, -1.0f }, 0.5f, rAI::material{ glm::vec3{ 0.0f, 0.0f, 1.0f }, glm::vec3{ 0.0f }, 0.0f } };

            rAI::sphere light = rAI::sphere{ glm::vec3{ 0.0f, 4.0f, -5.0f }, 2.f, rAI::material{ glm::vec3{ 0.0f }, glm::vec3{ 1.0f }, 50.0f } };

            spheres.push_back(sphere_ground);
            spheres.push_back(sphere_a);
            spheres.push_back(sphere_b);
            spheres.push_back(sphere_c);
            spheres.push_back(light);

            rAI::upload_scene(scene, spheres);
        }

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

