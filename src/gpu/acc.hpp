#pragma once

namespace nlcglib {


namespace acc {

template <class T, class U=T>
void
copy(T* target, const U* src, size_t n);

}  // acc
}  // nlcglib
