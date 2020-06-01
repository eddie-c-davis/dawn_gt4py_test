#ifndef DAWN_GT4PY_TEST_TIMER_CUDA_HPP
#define DAWN_GT4PY_TEST_TIMER_CUDA_HPP

namespace gridtools {

class timer_cuda {
protected:
  std::string name_;
  double total_time_ = 0.0;

public:
  timer_cuda(const std::string& name = "") : name_(name) {
    total_time_ = 0.0;
  }

  double total_time() {
    return total_time_;
  }

  void start() {
    total_time_ = 0.0;
  }

  void reset() {}

  void pause() {}
};

} // gridtools

#endif // DAWN_GT4PY_TEST_TIMER_CUDA_HPP
