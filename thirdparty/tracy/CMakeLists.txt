cmake_minimum_required(VERSION 3.18)
project(tracy LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Source files
set(SOURCES
    TracyClient.cpp
)

# Header files
set(HEADERS
    Tracy.hpp
    TracyC.h
    TracyOpenGL.hpp
    TracyVulkan.hpp
    TracyD3D11.hpp
    TracyD3D12.hpp
    TracyOpenCL.hpp
    TracyLua.hpp
)

# Create static library
add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/client
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/libbacktrace
)

# Configure Tracy based on build type
target_compile_definitions(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:>
    $<$<CONFIG:Release>:>
    $<$<CONFIG:Profile>:TRACY_ENABLE>
)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

# Find and link against threading library
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

# Install targets
install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${HEADERS}
    DESTINATION include/tracy
) 