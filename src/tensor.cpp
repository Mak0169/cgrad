#include "cgrad/tensor.h"

Tensor::Tensor(std::vector<float> data, std::vector<size_t> shape)
    : data_(std::move(data)), shape_(std::move(shape)) {}

size_t Tensor::numel() const {
    return data_.size();
}

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

float& Tensor::operator[](size_t i) {
    return data_[i];
}
