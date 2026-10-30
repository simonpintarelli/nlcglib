#include <nlcglib.hpp>
#include <nlcglib/interface.hpp>

void
nlcglib_consumer_run()
{
  nlcglib::initialize();
  nlcglib::finalize();
}
