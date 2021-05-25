#pragma once

#include <nlohmann/json.hpp>
#include <Kokkos_Complex.hpp>
#include <iostream>
#include <string>
#include <type_traits>
#include <type_traits>
#include "la/mvector.hpp"

namespace Kokkos {

template <class T>
void
to_json(nlohmann::json& j, const Kokkos::complex<T>& p)
{
  j = nlohmann::json{p.real(), p.imag()};
}
}  // namespace Kokkos


namespace nlcglib {

/// Store CG information in json.
class StepLogger
{
public:
  StepLogger(int i, std::string fname = "nlcg.json")
      : i(i), fname(fname)
  {
    dict["type"] = "cg_iteration";
    dict["step"] = i;
  }

  template<class X>
  std::enable_if_t<std::is_scalar<std::remove_reference_t<X>>::value> log(const std::string& key, X&& x);

  template<class X>
  void log(const std::string& key, const std::map<std::string, X>& v);

  template<class V>
  void log(const std::string& key, const mvector<V>& x);

  ~StepLogger()
  {
    std::ofstream fout(std::string("nlcg") + std::to_string(i) + ".json", std::ios_base::out);
    fout << dict;
    fout.flush();
  }

private:
  int i;
  std::string fname{"nlcg.json"};
  nlohmann::json dict;
};

template <class X>
std::enable_if_t<std::is_scalar<std::remove_reference_t<X>>::value>
StepLogger::log(const std::string& key, X&& x)
{
  dict[key] = x;
}

template<class X>
void StepLogger::log(const std::string& key, const std::map<std::string, X>& v)
{
  dict[key] = v;
}

template <class V>
void
StepLogger::log(const std::string& key, const mvector<V>& x)
{
  // assuming V is a 1-d kokkos array
  for (auto& elem : x) {
    auto x_key = elem.first;
    auto array = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), elem.second);
    std::vector<typename V::value_type> v(array.size());
    // std::vector<double> v(array.size());
    std::copy(array.data(), array.data() + array.size(), v.data());
    nlohmann::json entry;
    entry["ik"] = x_key.first;
    entry["ispn"] = x_key.second;
    entry["value"] = v;
    dict[key] += entry;
  }
}

}  // namespace nlcglib
