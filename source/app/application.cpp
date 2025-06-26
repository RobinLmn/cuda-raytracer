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

static constexpr int WINDOW_WIDTH = 1920;
static constexpr int WINDOW_HEIGHT = 1080;

static constexpr int RENDER_WIDTH = 1085;
static constexpr int RENDER_HEIGHT = 1026;

namespace 
{
    rAI::mesh make_quad(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, const rAI::material& material)
    {
        const glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

        const rAI::triangle t1{ {v0, v1, v2}, {normal, normal, normal} };
        const rAI::triangle t2{ {v0, v2, v3}, {normal, normal, normal} };

        const glm::vec3 min = glm::min(glm::min(v0, v1), glm::min(v2, v3));
        const glm::vec3 max = glm::max(glm::max(v0, v1), glm::max(v2, v3));

        rAI::mesh mesh;
        mesh.triangles = { t1, t2 };
        mesh.material = material;
        mesh.bounding_box = rAI::aabb{ min, max };

        return mesh;
    }

    std::vector<rAI::mesh> make_cornell_box()
    {
        std::vector<rAI::mesh> meshes(7);

        const rAI::material white_material{ glm::vec3{ 1.f }, glm::vec3{ 0.f }, 0.f, 0.f, glm::vec3{ 1.f }, 0.f };
        const rAI::material red_material{ glm::vec3{ 0.9f, 0.1f, 0.1f }, glm::vec3{ 0.f }, 0.f, 0.f, glm::vec3{ 1.f }, 0.f };
        const rAI::material green_material{ glm::vec3{ 0.1f, 0.9f, 0.1f }, glm::vec3{ 0.f }, 0.f, 0.f, glm::vec3{ 1.f }, 0.f };
        const rAI::material light_material{ glm::vec3{ 0.f }, glm::vec3{ 1.f, 1.f, 1.f }, 20.f, 0.f, glm::vec3{ 1.f }, 0.f };

        meshes.push_back(make_quad(glm::vec3{-2.f, 0.f, 2.f}, glm::vec3{2.f, 0.f, 2.f}, glm::vec3{2.f, 0.f, -2.f}, glm::vec3{-2.f, 0.f, -2.f}, white_material)); // Floor
        meshes.push_back(make_quad(glm::vec3{-2.f, 4.f, -2.f}, glm::vec3{2.f, 4.f, -2.f}, glm::vec3{2.f, 4.f, 2.f}, glm::vec3{-2.f, 4.f, 2.f}, white_material)); // Ceiling
        meshes.push_back(make_quad(glm::vec3{-2.f, 0.f, -2.f}, glm::vec3{2.f, 0.f, -2.f}, glm::vec3{2.f, 4.f, -2.f}, glm::vec3{-2.f, 4.f, -2.f}, white_material)); // Back
        meshes.push_back(make_quad(glm::vec3{-2.f, 0.f, 2.f}, glm::vec3{-2.f, 0.f, -2.f}, glm::vec3{-2.f, 4.f, -2.f}, glm::vec3{-2.f, 4.f, 2.f}, red_material)); // Left
        meshes.push_back(make_quad(glm::vec3{2.f, 0.f, -2.f}, glm::vec3{2.f, 0.f, 2.f}, glm::vec3{2.f, 4.f, 2.f}, glm::vec3{2.f, 4.f, -2.f}, green_material)); // Right
        meshes.push_back(make_quad(glm::vec3{2.f, 0.f, 2.f}, glm::vec3{-2.f, 0.f, 2.f}, glm::vec3{-2.f, 4.f, 2.f}, glm::vec3{2.f, 4.f, 2.f}, white_material)); // Front
        meshes.push_back(make_quad(glm::vec3{-0.5f, 3.99f, -0.5f}, glm::vec3{0.5f, 3.99f, -0.5f}, glm::vec3{0.5f, 3.99f, 0.5f}, glm::vec3{-0.5f, 3.99f, 0.5f}, light_material)); // Light

        return meshes;
    }

    std::vector<rAI::mesh> make_bed()
    {
        const rAI::material bed_material{ glm::vec3{ 1.f }, glm::vec3{ 0.f }, 0.f, 0.f, glm::vec3{ 1.f }, 0.f };
        const std::vector<rAI::mesh>& meshes = rAI::load_meshes_from_obj("../content/bed.obj", bed_material);

        return meshes;
    }

    std::vector<rAI::sphere> make_spheres()
    {
        const rAI::material sphere_material_a{ glm::vec3{ 0.95f, 0.95f, 0.95f }, glm::vec3{ 0.f }, 0.f, 1.f, glm::vec3{ 1.f }, 0.2f };
        const rAI::material sphere_material_b{ glm::vec3{ 0.8f, 0.2f, 0.2f }, glm::vec3{ 0.f }, 0.0f, 0.f, glm::vec3{ 1.f }, 0.f };
        const rAI::material sphere_material_c{ glm::vec3{ 0.2f, 0.6f, 0.9f }, glm::vec3{ 0.f }, 0.0f, 0.2f, glm::vec3{ 1.f }, 0.2f };

        std::vector<rAI::sphere> spheres;

        spheres.push_back(rAI::sphere{ glm::vec3{ 0.f, 1.f, 0.f }, 0.5f, sphere_material_a });
        spheres.push_back(rAI::sphere{ glm::vec3{ -1.2f, 1.f, 0.f }, 0.5f, sphere_material_b });
        spheres.push_back(rAI::sphere{ glm::vec3{ 1.2f, 1.f, 0.f }, 0.5f, sphere_material_c });

        return spheres;
    }

    rAI::rendering_context get_rendering_context(const app::camera& camera, const app::render_settings_data& render_settings_data)
    {
        return rAI::rendering_context
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
            render_settings_data.exposure,
            render_settings_data.gamma
        };
    }

    app::render_settings_data get_default_render_settings_data()
    {
        app::render_settings_data render_settings_data;
        render_settings_data.max_bounces = 5;
        render_settings_data.rays_per_pixel = 5;
        render_settings_data.diverge_strength = 2.f;
        render_settings_data.defocus_strength = 0.f;
        render_settings_data.focus_distance = 5.f;
        render_settings_data.exposure = 1.0f;
        render_settings_data.gamma = 2.2f;
        render_settings_data.sky_box.is_hidden = false;
        render_settings_data.sky_box.horizon_color = glm::vec3{ 1.f, 1.f, 1.f },
        render_settings_data.sky_box.zenith_color = glm::vec3{ .1f, 0.2f, 0.8f },
        render_settings_data.sky_box.ground_color = glm::vec3{ 0.9f };
        render_settings_data.sky_box.sun_direction = glm::normalize(glm::vec3{ 1.f, 1.f, -1.f });
        render_settings_data.sky_box.sun_intensity = 4.0f;
        render_settings_data.sky_box.sun_focus = 400.0f;
        render_settings_data.sky_box.brightness = 1.0f;
        render_settings_data.is_rendering = false;

        return render_settings_data;
    }

    app::camera get_default_camera()
    {
        const float near_plane = 0.1f;
        const float far_plane = 100.f;
        const float fov = 45.f;
        const float aspect_ratio = static_cast<float>(RENDER_WIDTH) / static_cast<float>(RENDER_HEIGHT);
        const glm::vec3 position{ 0.f, 1.f, 3.85f };
        const glm::vec3 direction{ 0.f, 0.f, -1.f };
        const float speed = 3.f;
        const float sensitivity = 0.005f;

        return app::camera{ near_plane, far_plane, fov, aspect_ratio, position, direction, speed, sensitivity };
    }

    rAI::scene make_scene()
    {
        rAI::scene scene;

        const std::vector<rAI::mesh>& meshes = make_cornell_box();
        const std::vector<rAI::sphere>& spheres = make_spheres();
            
        rAI::upload_scene(scene, spheres, meshes);

        return scene;
    }
}

namespace app
{
    void application::run()
    {
        core::window window{ WINDOW_WIDTH, WINDOW_HEIGHT, "rAI: GPU-Accelerated Raytracer" };
        core::editor editor;
        rAI::raytracer raytracer{ RENDER_WIDTH, RENDER_HEIGHT };
        
        render_settings_data render_settings_data = get_default_render_settings_data();
        camera camera = get_default_camera();

        const rAI::scene& scene = make_scene();

        const auto on_save_image = [&raytracer]()
        {
            const std::string& filename = core::new_file_dialog("render.png", "../docs", "Png Files\0*.png\0");

            const rAI::texture& render_texture = raytracer.get_render_texture();
            const std::vector<unsigned char>& pixels = render_texture.read_pixels();

            stbi_write_png(filename.c_str(), render_texture.get_width(), render_texture.get_height(), 4, pixels.data(), render_texture.get_width() * 4);
        };

        const auto on_render_stop = [&raytracer]()
        {
            raytracer.reset_accumulation();
        };

        editor.add_widget<viewport>(raytracer.get_render_texture().get_id(), RENDER_WIDTH, RENDER_HEIGHT);
        editor.add_widget<render_settings>(render_settings_data, on_save_image, on_render_stop);

        const std::chrono::high_resolution_clock clock{};
        std::chrono::high_resolution_clock::time_point last_time = clock.now();

        while (window.is_open())
        {
            float delta_time = std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1>>>(clock.now() - last_time).count();
            last_time = clock.now();

            window.update();

            if (!render_settings_data.is_rendering)
                camera.update(delta_time);

            raytracer.render(get_rendering_context(camera, render_settings_data), scene, render_settings_data.is_rendering);

            editor.render();
        }
    }
}

