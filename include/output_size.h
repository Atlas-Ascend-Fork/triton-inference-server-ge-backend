/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#ifndef OUTPUT_SIZE_H
#define OUTPUT_SIZE_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace triton::backend::npu_ge {

inline bool CheckedMultiplyOutputSize(int64_t value, int64_t factor, int64_t &result) noexcept
{
    if (value < 0 || factor < 0 || (factor != 0 && value > std::numeric_limits<int64_t>::max() / factor)) {
        return false;
    }
    result = value * factor;
    return true;
}

inline bool CalculateOutputByteSizes(std::size_t element_size, const std::vector<int64_t> &dimensions,
                                     bool include_batch, int64_t batch_total, std::size_t line_start,
                                     int64_t &total_bytes, int64_t &line_bytes) noexcept
{
    if (element_size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) || line_start > dimensions.size() ||
        (include_batch && batch_total < 0)) {
        return false;
    }

    int64_t checked_total = static_cast<int64_t>(element_size);
    if (include_batch && !CheckedMultiplyOutputSize(checked_total, batch_total, checked_total)) {
        return false;
    }
    for (const int64_t dimension : dimensions) {
        if (!CheckedMultiplyOutputSize(checked_total, dimension, checked_total)) {
            return false;
        }
    }

    int64_t checked_line = static_cast<int64_t>(element_size);
    for (std::size_t i = line_start; i < dimensions.size(); ++i) {
        if (!CheckedMultiplyOutputSize(checked_line, dimensions[i], checked_line)) {
            return false;
        }
    }

    if (static_cast<std::uint64_t>(checked_total) > std::numeric_limits<std::size_t>::max() ||
        static_cast<std::uint64_t>(checked_line) > std::numeric_limits<std::size_t>::max()) {
        return false;
    }

    total_bytes = checked_total;
    line_bytes = checked_line;
    return true;
}

} // namespace triton::backend::npu_ge

#endif
