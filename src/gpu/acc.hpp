#pragma once

namespace nlcglib {

int num_devices();

template<typename T>
void copy(T* target, const T* src, size_t n);

}  // nlcglib
