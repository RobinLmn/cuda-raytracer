#pragma once

namespace core
{
    class widget
    {
    public:
        virtual ~widget() = default;
        virtual void draw() = 0;
    };
}
