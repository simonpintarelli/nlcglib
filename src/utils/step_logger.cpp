#include "step_logger.h"
#include <Kokkos_Complex.hpp>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
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
namespace {

/// Store CG information in json.
class StepLogger
{
public:
  StepLogger(int i, std::string fname = "nlcg.json", bool active = true)
      : fname(fname)
      , active(active)
  {
    dict["type"] = "cg_iteration";
    dict["step"] = i;
  }

  template <class X>
  std::enable_if_t<std::is_scalar<std::remove_reference_t<X>>::value> log(const std::string& key,
                                                                          X&& x);

  template <class X>
  void log(const std::string& key, const std::map<std::string, X>& v);

  template <class V>
  void log(const std::string& key, const mvector<V>& x);

  void log(const std::string& key, const mvector<double>& x);

  ~StepLogger()
  {
    if (active) {
      std::ofstream fout(std::string("nlcg") + ".json", std::ios_base::app);
      fout << dict.dump();
      fout.flush();
    }
  }

private:
  std::string fname{"nlcg.json"};
  bool active;
  nlohmann::json dict;
};

template <class X>
std::enable_if_t<std::is_scalar<std::remove_reference_t<X>>::value>
StepLogger::log(const std::string& key, X&& x)
{
  if (!active) return;
  dict[key] = x;
}

template <class X>
void
StepLogger::log(const std::string& key, const std::map<std::string, X>& v)
{
  if (!active) return;
  dict[key] = v;
}

template <class V>
void
StepLogger::log(const std::string& key, const mvector<V>& x)
{
  if (!active) return;
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


}  // namespace

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
                   int freq)
{
  StepLogger logger(step, "nlcg.json", commk.rank() == 0);
  logger.log("F", free_energy);
  logger.log("EKS", ks_energy);
  logger.log("entropy", entropy);
  logger.log("slope_x", slope_x);
  logger.log("slope_eta", slope_eta);
  logger.log("fermi_energy", efermi);
  logger.log("ks_energy_comps", energy_components);
  if (step == 0) {
    logger.log("wk", wk);
  }

  if (step % freq == 0) {
    auto ek_host =
        eval_threaded(tapply(
                          [](auto&& x) {
                            return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
                          },
                          ek))
            .allgather(commk);

    auto fn_host =
        eval_threaded(tapply(
                          [](auto&& x) {
                            return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
                          },
                          fn))
            .allgather(commk);

    logger.log("eta", ek_host);
    logger.log("fn", fn_host);
  }
}


}  // namespace nlcglib
