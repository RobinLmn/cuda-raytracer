#include "texture.hpp"

#include <glad/glad.h>
#include <vector>

namespace rAI
{
    texture::texture(const int width, const int height)
        : id{ 0 }
        , unit{ 0 }
    {
        glActiveTexture(GL_TEXTURE0 + unit);

        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    texture::~texture()
    {
        glDeleteTextures(1, &id);
    }
    
    void texture::bind() const
    {
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, id);
    }

    void texture::unbind() const
    {
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    unsigned int texture::get_id() const
    {
        return id;
    }

    unsigned int texture::get_unit() const
    {
        return unit;
    }
}
