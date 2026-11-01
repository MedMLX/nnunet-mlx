"""Shared double-single Metal arithmetic for reference-precision operations.

Each value is stored as the unevaluated sum of two float32 components. These
operations approximate higher-precision sums and products; they are not IEEE
float64 arithmetic or a general-purpose double-precision implementation.
"""

DOUBLE_SINGLE_HEADER = r"""
struct DoubleSingle {
    float high;
    float low;
};

inline DoubleSingle ds_add(DoubleSingle left, DoubleSingle right) {
    float sum = left.high + right.high;
    float virtual_right = sum - left.high;
    float error = (left.high - (sum - virtual_right)) + (right.high - virtual_right);
    error += left.low + right.low;
    float high = sum + error;
    return {high, error - (high - sum)};
}

inline DoubleSingle ds_multiply(DoubleSingle left, DoubleSingle right) {
    float product = left.high * right.high;
    float error = metal::fma(left.high, right.high, -product);
    error += left.high * right.low + left.low * right.high;
    error += left.low * right.low;
    float high = product + error;
    return {high, error - (high - product)};
}
"""
