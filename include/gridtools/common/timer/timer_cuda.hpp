#ifndef DAWN_GT4PY_TEST_TIMER_CUDA_HPP
#define DAWN_GT4PY_TEST_TIMER_CUDA_HPP

namespace gridtools {
namespace dawn {

class timer_cuda {
protected:
  std::string name_;
  double total_time_ = 0.0;

public:
  timer_cuda(const std::string &name = "") : name_(name) {}

  double total_time() { return total_time_; }

  void start() { total_time_ = 0.0; }

  void reset() {}

  void pause() {}
};

} // namespace dawn
} // namespace gridtools

#endif // DAWN_GT4PY_TEST_TIMER_CUDA_HPP
