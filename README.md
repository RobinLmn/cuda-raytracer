# rAI: A GPU-Accelerated Raytracer

rAI is a real-time CUDA-powered raytracer featuring progressive rendering with real-time preview, anti-aliasing, and depth of field effects. It supports sphere and mesh rendering, and offers a variety of material types including diffuse, specular, glossy, and emissive surfaces. The renderer includes sky lighting with sun and ground reflection, along with HDR rendering capabilities.

<div align="center">
<img src="docs/tool_window.png" width="600">
<br>
<i>The tool window</i>
</div>

## Getting Started

### Prerequisites
- CUDA Toolkit 11.0 or later
- CMake 3.15 or later
- A CUDA-capable GPU
- C++17 compatible compiler

### Download

Download on Windows: [Link](https://github.com/robinlmn/DebrisDiskSimulation/releases/latest)

### Building the Project
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## The Journey

### CUDA-OpenGL interop

The raytracer is built on CUDA-OpenGL interoperability, allowing us to render directly to an OpenGL texture. This setup allows us to:

- Create an OpenGL texture that can be displayed as an ImGui::Image
- Map this texture to CUDA memory for direct writing
- Update the display in real-time as the raytracer progresses

### Basic Raytracing

The project began with implementing the fundamental components of a raytracer:
- Ray generation from the camera
- Ray-sphere intersection testing
- Basic material properties (diffuse color)
- Simple lighting calculations

<div align="center">
<img src="docs/basic_raytracing.png" width="300">
<br>
<i>The initial implementation showing basic sphere rendering</i>
</div>

### Progressive Rendering

To improve image quality and reduce noise, I implemented progressive rendering with an accumulation buffer. This allows the image to gradually refine over time.

<div align="center">
<table>
<tr>
<td align="center">
<img src="docs/without_progressive_rendering.png" width="300">
<br>
<i>Before progressive rendering - noisy image</i>
</td>
<td align="center">
<img src="docs/with_progressive_rendering.png" width="300">
<br>
<i>After progressive rendering - much cleaner result</i>
</td>
</tr>
</table>
</div>

### Anti-aliasing and Depth of Field

To reduce jagged edges and create more realistic renders, I implemented multi-sampling anti-aliasing. This technique shoots multiple rays per pixel with a small jitter from the camera point and averages their results. This creates smoother edges and reduces visual artifacts.

<div align="center">
<img src="docs/anti_aliasing.png" width="300">
<br>
<i>Anti-aliasing in action - notice the smoother edges</i>
</div>

For depth of field, I implemented a physical camera model that simulates how real cameras focus light. By adjusting the focus distance and defocus strength, we can create realistic bokeh effects where objects at different distances appear sharp or blurred.

<div align="center">
<table>
<tr>
<td align="center">
<img src="docs/focus_back.png" width="250">
<br>
<i>Focus on background</i>
</td>
<td align="center">
<img src="docs/focus.png" width="250">
<br>
<i>Focus on the center</i>
</td>
<td align="center">
<img src="docs/focus_front.png" width="250">
<br>
<i>Focus on foreground</i>
</td>
</tr>
</table>
</div>

### Mesh Support

The next major addition was support for complex geometry through triangle meshes:
- Triangle intersection testing
- OBJ file loading
- Bounding box optimization for performance

<div align="center">
<table>
<tr>
<td align="center">
<img src="docs/triangle.png" width="300">
<br>
<i>Basic triangle rendering</i>
</td>
<td align="center">
<img src="docs/mesh.png" width="300">
<br>
<i>Complex mesh loaded from OBJ file</i>
</td>
</tr>
</table>
</div>

### Advanced Materials

To create more realistic surfaces, I implemented:
- Specular reflection
- Glossy surfaces with adjustable smoothness
- Emissive materials for light sources

<div align="center">
<img src="docs/cornell_box_spheres.png" width="300">
<br>
<i>Showcasing different material types in the Cornell box scene</i>
</div>

### Tone Mapping

To handle high dynamic range rendering properly:
- Exposure control
- Gamma correction
- Tone mapping (reinhard / aces)

<div align="center">
<table>
<tr>
<td align="center">
<img src="docs/gamma_correction_reinhard_tone_mapping_exposure.png" width="300">
<br>
<i>Reinhard tone mapping</i>
</td>
<td align="center">
<img src="docs/aces_tone_mapping.png" width="300">
<br>
<i>ACES tone mapping</i>
</td>
</tr>
</table>
</div>



## License
This project is licensed under the MIT License - see the LICENSE file for details.
