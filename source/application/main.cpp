
#include "config.hpp"
#include "logger.hpp"
#include "math/vector.hpp"


void print_res(Vector& v)
{
    Logger lg(logging::CONSOLE);
    for (int i = 0; i < v.size() && i < 10; i++)
    {
        lg.log(
                std::string("res[") + std::to_string(i) + std::string("] = ")
                + std::to_string(v.at(i))
                );
    }
    lg.log("");
}


int main(int argc, char* argv[])
{
    Logger lg(logging::CONSOLE);
    lg.debug("The program was launched.");

    int N = 1000000;

    lg.log("Create VectorCPU...");
    Vector a(N, Device::CPU);
    Vector b(N, Device::CPU);
    Vector c(N, Device::CPU);

    lg.log("Create VectorGPU...");
    Vector d(N, Device::GPU);
    Vector e(N, Device::GPU);
    Vector f(N, Device::GPU);

    lg.log("Initializing vectors...");
    for (int i = 0; i < N; i++)
    {
        a.set_at(i, i);
        b.set_at(i, i+1);
        d.set_at(i, i);
        e.set_at(i, i+1);
    }

    lg.log("Calculate VectorGPU");
    f = d - e;
    lg.log(std::string("ok: ") + std::to_string(f.at(0)));

    lg.log("Calculate VectorCPU");
    c = a - b;
    lg.log(std::string("ok: ") + std::to_string(c.at(0)));

    lg.log("");
    lg.log("VectorCPU");
    print_res(c);
    lg.log("VectorGPU");
    print_res(f);

    return 0;
}

