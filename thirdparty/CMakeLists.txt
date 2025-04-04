cmake_minimum_required(VERSION 3.18)

# Add all third-party libraries
add_subdirectory(glad)
add_subdirectory(glfw)
add_subdirectory(imgui)
add_subdirectory(tracy)

# Set folder organization for Visual Studio
set_target_properties(glad PROPERTIES FOLDER "ThirdParty")
set_target_properties(glfw PROPERTIES FOLDER "ThirdParty")
set_target_properties(imgui PROPERTIES FOLDER "ThirdParty")
set_target_properties(tracy PROPERTIES FOLDER "ThirdParty") 