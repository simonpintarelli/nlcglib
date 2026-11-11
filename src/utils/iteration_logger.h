#pragma once

#include "mvp2/slope.h"

namespace nlcglib {

struct nlcg_info;

nlcg_info
print_info(
    double free_energy, double ks_energy, double entropy, slope_t slope, double efermi, int step);
void
log_restart_warning(int iteration, slope_t slope);
void
log_restart(int iteration, int interval);

}  // namespace nlcglib
