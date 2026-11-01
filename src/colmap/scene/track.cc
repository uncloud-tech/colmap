// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/track.h"

namespace colmap {

Track::Track() {}

Track::Track(std::shared_ptr<MappingLedger> ledger)
    : mapping_credit_(ledger ? ledger->Acquire(0) : nullptr) {}

Track::Track(const Track& other)
    : mapping_credit_(other.mapping_credit_
                          ? other.mapping_credit_->ledger->Acquire(
                                MappingMultiply(other.Length(),
                                                sizeof(TrackElement)))
                          : nullptr),
      elements_(other.elements_) {}

Track& Track::operator=(const Track& other) {
  if (this != &other) {
    if (!mapping_credit_ && !other.mapping_credit_) {
      // Preserve legacy vector assignment/capacity reuse. TrackElement copying
      // cannot throw; any required allocation precedes replacement of storage.
      elements_ = other.elements_;
    } else {
      Track replacement(other);
      Swap(replacement);
    }
  }
  return *this;
}

Track& Track::operator=(Track&& other) noexcept {
  if (this != &other) {
    Track replacement(std::move(other));
    Swap(replacement);
  }
  return *this;
}

void Track::Swap(Track& other) noexcept {
  static_assert(noexcept(elements_.swap(other.elements_)));
  mapping_credit_.swap(other.mapping_credit_);
  elements_.swap(other.elements_);
}

void Track::ReplaceCapacity(const size_t capacity) {
  if (capacity > elements_.max_size()) {
    throw std::length_error("mapping track shape");
  }
  // Charge the requested payload before allocating. Any allocator-provided
  // excess remains UNKNOWN; there is deliberately no postallocation Acquire.
  auto credit = mapping_credit_->ledger->Acquire(
      MappingMultiply(capacity, sizeof(TrackElement)));
  {
    std::vector<TrackElement> replacement;
    replacement.reserve(capacity);
    replacement.insert(replacement.end(), elements_.begin(), elements_.end());
    elements_.swap(replacement);
  }  // Old storage dies while both tickets are still held.
  mapping_credit_.swap(credit);
}

void Track::PrepareAppend(const size_t count) {
  if (!mapping_credit_) return;
  if (count > elements_.max_size() - elements_.size()) {
    throw std::length_error("mapping track shape");
  }
  const size_t required = elements_.size() + count;
  if (required > elements_.capacity()) {
    const size_t doubled = elements_.capacity() > elements_.max_size() / 2
                               ? elements_.max_size()
                               : 2 * elements_.capacity();
    ReplaceCapacity(std::max(required, doubled));
  }
}

TrackElement::TrackElement()
    : image_id(kInvalidImageId), point2D_idx(kInvalidPoint2DIdx) {}

TrackElement::TrackElement(const image_t image_id, const point2D_t point2D_idx)
    : image_id(image_id), point2D_idx(point2D_idx) {}

void Track::DeleteElement(const image_t image_id, const point2D_t point2D_idx) {
  elements_.erase(
      std::remove_if(elements_.begin(),
                     elements_.end(),
                     [image_id, point2D_idx](const TrackElement& element) {
                       return element.image_id == image_id &&
                              element.point2D_idx == point2D_idx;
                     }),
      elements_.end());
}

std::ostream& operator<<(std::ostream& stream, const TrackElement& track_el) {
  stream << "TrackElement(image_id=" << track_el.image_id
         << ", point2D_idx=" << track_el.point2D_idx << ")";
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const Track& track) {
  stream << "Track(elements=[";
  for (auto it = track.Elements().begin(); it != track.Elements().end();) {
    stream << *it;
    if (++it != track.Elements().end()) {
      stream << ", ";
    }
  }
  stream << "])";
  return stream;
}

}  // namespace colmap
