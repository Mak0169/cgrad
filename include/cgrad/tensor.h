#pragma once
#include <vector>
#include <cstddef>

class Tensor {
public:
    Tensor(std::vector<float> data, std::vector<size_t> shape);

    size_t numel() const;
    const std::vector<size_t>& shape() const;
    float& operator[](size_t i);

private:
    std::vector<float> data_;
    std::vector<size_t> shape_;
};
