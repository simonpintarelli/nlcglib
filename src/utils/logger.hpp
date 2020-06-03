#pragma once

#include <mpi.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <list>
#include <memory>
#include <mutex>
#include <string>

#include "csingleton.hpp"

namespace nlcglib {
class Logger : public CSingleton<Logger>
{
public:
  Logger() {
    MPI_Comm_rank(MPI_COMM_WORLD, &pid_);
  }

  void attach_file(const std::string& prefix = "out", const std::string& suffix = ".log")
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &pid_);
    stream_ptr_ = std::make_shared<std::ofstream>(prefix + std::to_string(pid_) + suffix);
  }

  /// only master rank writes
  void attach_file_master(const std::string& fname = "nlcg.out")
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &pid_);
    if (pid_ == 0)
      stream_ptr_ = std::make_shared<std::ofstream>(fname);
  }

  template <typename T>
  Logger& operator<<(const T& output)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    sbuf_.str("");

    for (auto& v : prefixes_) {
      sbuf_ << v << "::";
    }
    sbuf_ << output;
    // output to console & file
    if ((bool)stream_ptr_) {
      auto& out = *(stream_ptr_.get());
      out << sbuf_.str();
    }
    if (!detach_stdout_ && pid_ == 0) std::cout << sbuf_.str();

    return *this;
  }

  void push_prefix(const std::string& tag)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    prefixes_.push_back(tag);
  }

  void pop_prefix()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    prefixes_.pop_back();
  }

  void clear_prefix()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    prefixes_.clear();
  }

  void flush()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // make sure to write intermediate changes
    auto& out = *(stream_ptr_.get());
    out.flush();
  }

  void detach_stdout() { detach_stdout_ = true; }

  void attach_stdout() { detach_stdout_ = false; }

  bool is_detached() { return detach_stdout_ == true; }

private:
  std::list<std::string> prefixes_;
  std::shared_ptr<std::ostream> stream_ptr_;
  std::mutex mutex_;
  std::stringstream sbuf_;
  bool detach_stdout_ = false;
  int pid_ = 0;
};

}  // namespace nlcglib
