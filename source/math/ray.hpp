#pragma once

#include "math/vec3.hpp"

#include <cuda_runtime.h>

namespace rAI
{
    struct ray
    {
        vec3 origin;
        vec3 direction;
    };
}
