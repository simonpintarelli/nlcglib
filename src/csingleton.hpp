#pragma once

#include <memory>
#include <mutex>


namespace nlcglib {

template <typename T>
class CSingleton
{
public:
  static T& GetInstance();

private:
  static std::unique_ptr<T> m_instance;
  static std::once_flag m_onceFlag;
  CSingleton(void) {}
  friend T;
};

template <typename T>
std::unique_ptr<T> CSingleton<T>::m_instance;

template <typename T>
std::once_flag CSingleton<T>::m_onceFlag;

template <typename T>
inline T&
CSingleton<T>::GetInstance()
{
  std::call_once(m_onceFlag, [] { m_instance.reset(new T); });
  return *m_instance.get();
}

}  // namespace nlcglib
