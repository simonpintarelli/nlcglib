#include "step_logger.hpp"
#include <nlohmann/json.hpp>

namespace nlcglib {

void
StepLogger::log(const std::string& key, const mvector<double>& x)
{
  if (!active) return;
  // assuming V is a 1-d kokkos array
  for (auto& elem : x) {
    auto x_key = elem.first;
    // std::vector<double> v(array.size());
    nlohmann::json entry;
    entry["ik"] = x_key.first;
    entry["ispn"] = x_key.second;
    entry["value"] = elem.second;
    dict[key] += entry;
  }
}


}  // namespace nlcglib
