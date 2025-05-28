#include "application.hpp"

#include "core/log.hpp"
#include "core/window.hpp"
#include "core/editor.hpp"
#include "core/file_utility.hpp"

#include "raytracer/raytracer.hpp"
#include "raytracer/mesh_loader.hpp"

#include "app/camera.hpp"
#include "app/viewport.hpp"
#include "app/render_settings.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image/stb_image_write.h>

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
            // rAI::sphere sphere_a = rAI::sphere{ glm::vec3{ -1.2f, 0.5f, -1.0f }, 0.5f, rAI::material{ glm::vec3{ 0.9f, 0.1f, 0.3f }, glm::vec3{ 0.0f }, 0.0f, 0.f, glm::vec3{ 1.0f }, 0.f } };
            // rAI::sphere sphere_b = rAI::sphere{ glm::vec3{ -0.3f, 0.5f, -5.0f }, 0.5f, rAI::material{ glm::vec3{ 0.2f, 0.8f, 0.4f}, glm::vec3{ 0.0f }, 0.0f, 0.f, glm::vec3{ 1.0f }, 0.f } };
            // rAI::sphere sphere_c = rAI::sphere{ glm::vec3{ 1.2f, 0.5f, -1.0f }, 0.5f, rAI::material{ glm::vec3{ 0.4f, 0.2f, 0.9f }, glm::vec3{ 0.0f }, 0.0f, 0.f, glm::vec3{ 1.0f }, 0.f } };

            // rAI::sphere sphere_d = rAI::sphere{ glm::vec3{ -5.f, 0.5f, -10.0f }, 0.5f, rAI::material{ glm::vec3{ 0.9f, 0.6f, 0.1f }, glm::vec3{ 0.0f }, 0.0f, 0.f, glm::vec3{ 1.0f }, 0.f } };
            // rAI::sphere sphere_e = rAI::sphere{ glm::vec3{ 2.f, 0.5f, -10.0f }, 0.5f, rAI::material{ glm::vec3{ 0.1f, 0.7f, 0.8f }, glm::vec3{ 0.0f }, 0.0f, 0.f, glm::vec3{ 1.0f }, 0.f } };

            // std::vector<rAI::sphere> spheres;

            // spheres.push_back(sphere_a);
            // spheres.push_back(sphere_b);
            // spheres.push_back(sphere_c);
            // spheres.push_back(sphere_d);
            // spheres.push_back(sphere_e);

            rAI::material bed_material{ glm::vec3{ 0.8f, 0.8f, 0.8f }, glm::vec3{ 0.0f }, 0.0f, 0.0f, glm::vec3{ 1.0f }, 0.0f };
            std::vector<rAI::mesh> meshes = rAI::load_meshes_from_obj("../content/bed.obj", bed_material);

            rAI::upload_scene(scene, {}, meshes);
        }

        camera camera{ 0.1f, 100.0f, 45.f, 1085.f / 1026.f };
        
        editor.add_widget<viewport>(raytracer.get_render_texture().get_id());

        const auto clock = std::chrono::high_resolution_clock{};
        auto last_time = clock.now();

        render_settings_data render_settings_data;
        render_settings_data.max_bounces = 8;
        render_settings_data.rays_per_pixel = 5;
        render_settings_data.diverge_strength = 2.f;
        render_settings_data.defocus_strength = 0.f;
        render_settings_data.focus_distance = 5.f;
        render_settings_data.sky_box.is_hidden = false;
        render_settings_data.sky_box.horizon_color = glm::vec3(1.f, 1.f, 1.f),
        render_settings_data.sky_box.zenith_color = glm::vec3(0.36f, 0.58f, 0.8f),
        render_settings_data.sky_box.ground_color = glm::vec3(0.4f, 0.4f, 0.4f);
        render_settings_data.sky_box.sun_direction = glm::normalize(glm::vec3(1.0f, 1.0f, -1.0f));
        render_settings_data.sky_box.sun_intensity = 70.0f;
        render_settings_data.sky_box.sun_focus = 400.0f;
        render_settings_data.is_rendering = false;

        const auto on_save_image = [&raytracer]()
        {
            const std::string& filename = core::new_file_dialog("render.png", "../docs", "Png Files\0*.png\0");

            const std::vector<unsigned char>& pixels = raytracer.get_render_texture().read_pixels();
            stbi_write_png(filename.c_str(), 1085, 1026, 4, pixels.data(), 1085 * 4);
        };

        editor.add_widget<render_settings>(render_settings_data, on_save_image);

        bool was_rendering = render_settings_data.is_rendering;

        while (window.is_open())
        {
            using seconds = std::chrono::duration<float, std::ratio<1>>;
        
            float delta_time = std::chrono::duration_cast<seconds>(clock.now() - last_time).count();
            last_time = clock.now();

            window.update();

            if (!render_settings_data.is_rendering)
            {
                camera.update(delta_time);
            }

            if (was_rendering && !render_settings_data.is_rendering)
            {
                raytracer.reset_accumulation();
            }
            was_rendering = render_settings_data.is_rendering;

            rAI::rendering_context rendering_context
            { 
                camera.get_position(), 
                camera.get_inverse_view_matrix(), 
                camera.get_inverse_projection_matrix(),
                render_settings_data.max_bounces,
                render_settings_data.rays_per_pixel,
                render_settings_data.diverge_strength,
                render_settings_data.defocus_strength,
                render_settings_data.focus_distance,
                render_settings_data.sky_box,
            };
            
            raytracer.render(rendering_context, scene, render_settings_data.is_rendering);

            editor.render();
        }
    }
}

