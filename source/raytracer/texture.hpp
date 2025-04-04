#pragma once

namespace rAI
{
    class texture
    {
    public:
        texture(const int width, const int height);
        ~texture();

    public:
        void bind() const;
        void unbind() const;

        unsigned int get_id() const;
        unsigned int get_unit() const;

    private:
        unsigned int id;
        unsigned int unit;
    };
}
