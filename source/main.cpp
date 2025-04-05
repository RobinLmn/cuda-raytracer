#include "core/log.hpp"

#include "app/application.hpp"

int main()
{
    core::logger::initialize();

    app::application application;
    application.run();
}