#include <iostream>
#include "cgrad/tensor.h"

int main() {
    Tensor t({1, 2, 3, 4}, {2, 2});
    std::cout << t.numel() << "\n";
    t[0] = 10;
    return 0;
}
