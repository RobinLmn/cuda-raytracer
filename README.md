# GPU-Accelerated Raytracer

**rAI** is a real-time raytracer built with CUDA and OpenGL interoperability. It features progressive rendering, anti-aliasing, and depth of field effects. The engine supports both spheres and triangle meshes, and includes a physically-inspired material system with diffuse, specular, glossy, and emissive surfaces. Lighting includes a procedural sky with sun and ground gradients, and the final image is rendered in HDR with tone mapping.

<div align="center">
<img src="docs/tool_window.png" width="600"><br>
<i>The tool window</i>
</div>

## Getting Started

### Download

Download for Windows: [Download Link](https://github.com/robinlmn/rAI/releases/latest)  
*A CUDA-capable NVIDIA GPU is required.*

### Building Prerequisites

- CUDA Toolkit 11.0+
- CMake 3.15+
- C++20-compatible compiler
- NVIDIA GPU with CUDA support

## Key Features and Implementation

### CUDA-OpenGL Interoperability

The raytracer is built on CUDA-OpenGL interoperability, allowing us to render directly to an OpenGL texture. This setup allows us to:

- Create an OpenGL texture that can be displayed as an `ImGui::Image` in the viewport
- Map this texture to CUDA memory for direct writing
- Launch a CUDA kernel with a 2D grid matching the image dimensions for parallel pixel processing
- Write rendered pixels directly to GPU memory without CPU transfer

### Foundational Raytracing

The initial implementation focused on raytracing fundamentals:

- Camera-based ray generation
- Ray-sphere intersection
- Diffuse shading
- Simple lighting model

Camera controls (**WASD** and **right-click drag**) allow you to move around the scene and rotate the view. On the right panel, **Save** allows you to save the render as a `png` image.

<div align="center">
<img src="docs/basic_raytracing.png" width="300"><br>
<i>First pass: basic shading and intersection</i>
</div>

### Progressive Rendering

Progressive rendering with frame accumulation reduces noise and improves visual fidelity. Over time, the image converges to a clean result. Clicking **Render** on the right panel freezes the camera controls and starts the accumulation process. **Stop** stops the accumulation and resets it. It also unfreezes camera controls.

<div align="center">
<table>
<tr>
<td align="center"><img src="docs/without_progressive_rendering.png" width="300"><br><i>Without progressive rendering</i></td>
<td align="center"><img src="docs/with_progressive_rendering.png" width="300"><br><i>After progressive accumulation</i></td>
</tr>
</table>
</div>

### Anti-Aliasing & Depth of Field

Multi-sample anti-aliasing is achieved by jittering ray origins. Results are averaged to produce smooth edges. The **Diverge Strength** setting controls jitter intensity, acting like a blur.

<div align="center">
<img src="docs/anti_aliasing.png" width="300"><br>
<i>Multi-sample anti-aliasing</i>
</div>

Depth of field is implemented by perturbing ray directions across a virtual aperture and adjusting convergence at a configurable focal plane, producing realistic bokeh effects. This is controlled by the **Defocus Strength** and **Focus Distance**.

<div align="center">
<table>
<tr>
<td align="center"><img src="docs/focus_back.png" width="250"><br><i>Focus: background</i></td>
<td align="center"><img src="docs/focus.png" width="250"><br><i>Focus: mid</i></td>
<td align="center"><img src="docs/focus_front.png" width="250"><br><i>Focus: foreground</i></td>
</tr>
</table>
</div>

### Mesh Support

Mesh rendering was added to go beyond primitives:

- Ray-triangle intersection with backface culling
- OBJ file loading
- Bounding box acceleration for performance

<div align="center">
<table>
<tr>
<td align="center"><img src="docs/triangle.png" width="300"><br><i>Single triangle</i></td>
<td align="center"><img src="docs/mesh.png" width="300"><br><i>Imported mesh (OBJ)</i></td>
</tr>
</table>
</div>

### Physically-Inspired Materials

Materials include:
- Mirror-like specular reflection
- Glossy highlights with adjustable roughness
- Emissive surfaces to simulate area lights

<div align="center">
<img src="docs/cornell_box_spheres.png" width="300"><br>
<i>Cornell box with a variety of materials</i>
</div>

### Tone Mapping & HDR

To support HDR lighting:
- Exposure control
- Gamma correction
- Tone mapping (Reinhard and ACES)

<div align="center">
<table>
<tr>
<td align="center"><img src="docs/gamma_correction_reinhard_tone_mapping_exposure.png" width="300"><br><i>Reinhard tone mapping</i></td>
<td align="center"><img src="docs/aces_tone_mapping.png" width="300"><br><i>ACES tone mapping</i></td>
</tr>
</table>
</div>

## License

This project is released under the MIT License. See the LICENSE file for details.
