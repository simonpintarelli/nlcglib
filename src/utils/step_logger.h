#pragma once

#include <map>
#include <string>
#include "la/vector_types.h"

namespace nlcglib {

class Communicator;

void
cg_write_step_json(double free_energy,
                   double ks_energy,
                   double entropy,
                   double slope_x,
                   double slope_eta,
                   double efermi,
                   const mvector<host_vector_t>& ek,
                   const mvector<host_vector_t>& fn,
                   const mvector<double>& wk,
                   std::map<std::string, double> energy_components,
                   Communicator& commk,
                   int step,
                   int freq);

}  // namespace nlcglib
