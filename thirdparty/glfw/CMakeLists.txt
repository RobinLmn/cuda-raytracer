cmake_minimum_required(VERSION 3.18)

# Set CMP0115 policy to NEW to require explicit file extensions
cmake_policy(SET CMP0115 NEW)

project(glfw LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Configure glfw_config.h
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/glfw_config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/src/glfw_config.h
)

# Common source files
set(SOURCES
    src/context.c
    src/init.c
    src/input.c
    src/monitor.c
    src/vulkan.c
    src/window.c
)

# Platform-specific source files
if(WIN32)
    list(APPEND SOURCES
        src/win32_init.c
        src/win32_joystick.c
        src/win32_monitor.c
        src/win32_time.c
        src/win32_thread.c
        src/win32_window.c
        src/wgl_context.c
        src/egl_context.c
        src/osmesa_context.c
    )
    set(GLFW_PLATFORM_DEFINES _GLFW_WIN32 _CRT_SECURE_NO_WARNINGS)
else()
    list(APPEND SOURCES
        src/x11_init.c
        src/x11_monitor.c
        src/x11_window.c
        src/xkb_unicode.c
        src/posix_time.c
        src/posix_thread.c
        src/glx_context.c
        src/egl_context.c
        src/osmesa_context.c
        src/linux_joystick.c
    )
    set(GLFW_PLATFORM_DEFINES _GLFW_X11)
endif()

# Header files
set(HEADERS
    include/GLFW/glfw3.h
    include/GLFW/glfw3native.h
    ${CMAKE_CURRENT_BINARY_DIR}/src/glfw_config.h
)

# Create static library
add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/src  # Add this for glfw_config.h
)

# Platform-specific settings
target_compile_definitions(${PROJECT_NAME} PRIVATE ${GLFW_PLATFORM_DEFINES})

# Install targets
install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
) 