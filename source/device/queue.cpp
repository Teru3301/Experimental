
#include "device/queue.hpp"

sycl::queue& get_default_queue() {
    static sycl::queue q(sycl::gpu_selector_v);
    return q;
}

