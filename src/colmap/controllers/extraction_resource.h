// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace colmap {
inline uint64_t ExtractionAdd(uint64_t a, uint64_t b) {
  if (b > std::numeric_limits<uint64_t>::max() - a)
    throw std::overflow_error("extraction resource addition overflow");
  return a + b;
}
inline uint64_t ExtractionMultiply(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::overflow_error("extraction resource multiplication overflow");
  return a * b;
}

// Grayscale SIFT: actual decoded channels (alpha dropped), conversion overlap,
// original + resized pixels, gravity rotation's old + new pixels, and
// CPU SIFT's grayscale + float staging (5T). Staging is prepaid here.
// The maximum is held from the SAME decoder header until writer retirement.
// OIIO/VLFeat/SiftGPU internals and allocator bookkeeping remain UNKNOWN policy.
inline uint64_t ExtractionBitmapPayload(uint64_t width,
                                       uint64_t height,
                                       int file_channels,
                                       int max_size,
                                       bool cpu_staging = true) {
  if (!width || !height || file_channels < 1 || file_channels > 4 || max_size <= 0)
    throw std::invalid_argument("unsupported extraction bitmap header");
  const auto area = ExtractionMultiply(width, height);
  // Bitmap currently uses int multiplication even for RGB conversion.
  if (ExtractionMultiply(area, 3) > std::numeric_limits<int>::max())
    throw std::invalid_argument("extraction bitmap exceeds int payload range");
  const int channels = file_channels <= 2 ? 1 : 3;
  const auto decode = ExtractionMultiply(area, channels == 1 ? 1 : 4);
  uint64_t resized = area;
  const bool resize = std::max(width, height) > static_cast<uint64_t>(max_size);
  if (resize) {
    const double scale = static_cast<double>(max_size) / std::max(width, height);
    const auto w = std::max<uint64_t>(1, std::llround(width * scale));
    const auto h = std::max<uint64_t>(1, std::llround(height * scale));
    resized = ExtractionMultiply(w, h);
  }
  return std::max({decode, resize ? ExtractionAdd(area, resized) : area,
                   ExtractionMultiply(resized, cpu_staging ? 5 : 2)});
}

// A single frozen logical phase ledger, not an RSS/device/workspace cap.
// Only reader admission waits; growth ALWAYS refuses immediately, since waiting
// with live jobs could deadlock every extractor. Credits are move-independent
// owners travelling with ImageData, released only after its payload is freed.
class ExtractionBudget {
 private:
  struct State {
    uint64_t capacity, policy, reserve, used = 0, peak = 0;
    bool cancelled = false;
    std::mutex mutex;
    std::condition_variable changed;
    bool Fits(uint64_t bytes) const {
      const auto raw = ExtractionAdd(bytes, policy);
      const auto margin = ExtractionAdd(ExtractionMultiply(raw / 100, 3),
                                        (raw % 100 * 3 + 99) / 100);
      return ExtractionAdd(ExtractionAdd(raw, margin), reserve) <= capacity;
    }
  };
 public:
  class Ticket {
   public:
    ~Ticket() {
      std::lock_guard<std::mutex> lock(state_->mutex);
      state_->used -= bytes_;
      state_->changed.notify_all();
    }
    Ticket(const Ticket&) = delete;
    Ticket& operator=(const Ticket&) = delete;
    void Grow(uint64_t bytes) {
      std::lock_guard<std::mutex> lock(state_->mutex);
      const auto next = ExtractionAdd(state_->used, bytes);
      if (state_->cancelled || !state_->Fits(next))
        throw std::runtime_error(
            "extraction logical host growth refused; workers must drain "
            "(not a physical-fit guarantee)");
      state_->used = next;
      state_->peak = std::max(state_->peak, next);
      bytes_ = ExtractionAdd(bytes_, bytes);
    }
   private:
    friend class ExtractionBudget;
    explicit Ticket(std::shared_ptr<State> state) : state_(std::move(state)) {}
    std::shared_ptr<State> state_;
    uint64_t bytes_ = 0;
  };
  ExtractionBudget(uint64_t capacity, uint64_t policy, uint64_t reserve)
      : state_(std::make_shared<State>()) {
    state_->capacity = capacity;
    state_->policy = policy;
    state_->reserve = reserve;
  }
  std::shared_ptr<Ticket> Acquire(uint64_t bytes,
                                const std::function<bool()>& stopped) {
    auto ticket = std::shared_ptr<Ticket>(new Ticket(state_));
    std::unique_lock<std::mutex> lock(state_->mutex);
    if (!state_->Fits(bytes))
      throw std::runtime_error(
          "extraction logical host singleton refused before decode "
          "(not a physical-fit guarantee)");
    while (!state_->cancelled &&
           !state_->Fits(ExtractionAdd(state_->used, bytes))) {
      if (stopped()) throw std::runtime_error("extraction admission cancelled");
      // Signals are owned by real main, which cannot notify this CV.
      state_->changed.wait_for(lock, std::chrono::milliseconds(25));
    }
    if (state_->cancelled || stopped())
      throw std::runtime_error("extraction admission cancelled");
    state_->used = ExtractionAdd(state_->used, bytes);
    state_->peak = std::max(state_->peak, state_->used);
    ticket->bytes_ = bytes;
    return ticket;
  }
  void Cancel() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->cancelled = true;
    state_->changed.notify_all();
  }
  uint64_t Peak() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->peak;
  }
  uint64_t Used() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->used;
  }
 private:
  std::shared_ptr<State> state_;
};
}  // namespace colmap
