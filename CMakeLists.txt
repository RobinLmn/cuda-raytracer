cmake_minimum_required(VERSION 3.18)
project(rAI LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(thirdparty/GLFW)
add_subdirectory(thirdparty/glad)
add_subdirectory(thirdparty/tracy)
add_subdirectory(thirdparty/imgui)

find_package(CUDA REQUIRED)

include_directories(
    source
    thirdparty/
    thirdparty/GLFW/include
    thirdparty/glad/include
    thirdparty/spdlog/include
    thirdparty/glm
    thirdparty/imgui
    thirdparty/stb_image
    ${CUDA_INCLUDE_DIRS}
)

file(GLOB_RECURSE SOURCES 
    "source/*.cpp"
    "source/*.cu"
)

add_executable(${PROJECT_NAME} ${SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES 70)

target_link_libraries(rAI PRIVATE
    glfw
    glad
    tracy
    imgui
    ${CUDA_LIBRARIES}
)

target_compile_definitions(rAI PRIVATE
    _CRT_SECURE_NO_WARNINGS
    GLFW_INCLUDE_NONE
)

target_compile_options(rAI PRIVATE
    $<$<CONFIG:Debug>:-DDEBUG>
    $<$<CONFIG:Release>:-DRELEASE -O2>
)

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT rAI)

# Create virtual folders to make it look nicer in VS
if(MSVC_IDE)
	# Macro to preserve source files hierarchy in the IDE
	macro(GroupSources curdir)
		file(GLOB children RELATIVE ${PROJECT_SOURCE_DIR}/${curdir} ${PROJECT_SOURCE_DIR}/${curdir}/*)

		foreach(child ${children})
			if(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/${curdir}/${child})
				GroupSources(${curdir}/${child})
			else()
				string(REPLACE "/" "\\" groupname ${curdir})
				string(REPLACE "src" "Sources" groupname ${groupname})
				source_group(${groupname} FILES ${PROJECT_SOURCE_DIR}/${curdir}/${child})
			endif()
		endforeach()
	endmacro()

	# Run macro
	GroupSources(src)
endif()