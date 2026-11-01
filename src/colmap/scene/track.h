// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/scene/mapping_resource.h"
#include "colmap/util/logging.h"
#include "colmap/util/types.h"

#include <utility>
#include <vector>

namespace colmap {

// Track class stores all observations of a 3D point.
struct TrackElement {
  TrackElement();
  TrackElement(image_t image_id, point2D_t point2D_idx);
  // The image in which the track element is observed.
  image_t image_id;
  // The point in the image that the track element is observed.
  point2D_t point2D_idx;

  inline bool operator==(const TrackElement& other) const;
  inline bool operator!=(const TrackElement& other) const;
};

class Track {
 public:
  Track();
  // Component-only opt-in. Native mapping does not attach Track budgets yet.
  // Mutable Elements() growth and externally allocated SetElements arguments
  // are outside this interception boundary. Allocator excess is UNKNOWN.
  explicit Track(std::shared_ptr<MappingLedger> ledger);
  Track(const Track& other);
  Track(Track&& other) noexcept = default;
  Track& operator=(const Track& other);
  Track& operator=(Track&& other) noexcept;

  // The number of track elements.
  inline size_t Length() const;

  // Access all elements.
  inline const std::vector<TrackElement>& Elements() const;
  inline std::vector<TrackElement>& Elements();
  inline void SetElements(std::vector<TrackElement> elements);

  // Access specific elements.
  inline const TrackElement& Element(size_t idx) const;
  inline TrackElement& Element(size_t idx);
  inline void SetElement(size_t idx, const TrackElement& element);

  // Append new elements.
  inline void AddElement(const TrackElement& element);
  inline void AddElement(image_t image_id, point2D_t point2D_idx);
  inline void AddElements(const std::vector<TrackElement>& elements);

  // Delete existing element.
  inline void DeleteElement(size_t idx);
  void DeleteElement(image_t image_id, point2D_t point2D_idx);

  // Requests that the track capacity be at least enough to contain the
  // specified number of elements.
  inline void Reserve(size_t num_elements);

  // Shrink the capacity of track vector to fit its size to save memory.
  inline void Compress();

  inline bool operator==(const Track& other) const;
  inline bool operator!=(const Track& other) const;

 private:
  void Swap(Track& other) noexcept;
  void ReplaceCapacity(size_t capacity);
  void PrepareAppend(size_t count);

  // Declaration order keeps credit alive until its vector has been destroyed.
  std::shared_ptr<MappingLedger::Ticket> mapping_credit_;
  std::vector<TrackElement> elements_;
};

std::ostream& operator<<(std::ostream& stream, const TrackElement& track_el);
std::ostream& operator<<(std::ostream& stream, const Track& track);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

bool TrackElement::operator==(const TrackElement& other) const {
  return image_id == other.image_id && point2D_idx == other.point2D_idx;
}

bool TrackElement::operator!=(const TrackElement& other) const {
  return !(*this == other);
}

size_t Track::Length() const { return elements_.size(); }

const std::vector<TrackElement>& Track::Elements() const { return elements_; }

std::vector<TrackElement>& Track::Elements() { return elements_; }

void Track::SetElements(std::vector<TrackElement> elements) {
  if (!mapping_credit_) {
    elements_ = std::move(elements);
    return;
  }
  // The incoming vector was allocated by the caller (UNKNOWN). Prepare an
  // independently owned replacement before changing this Track.
  Track replacement(mapping_credit_->ledger);
  replacement.Reserve(elements.size());
  replacement.elements_.insert(
      replacement.elements_.end(), elements.begin(), elements.end());
  Swap(replacement);
}

// Access specific elements.
const TrackElement& Track::Element(const size_t idx) const {
  return elements_.at(idx);
}

TrackElement& Track::Element(const size_t idx) { return elements_.at(idx); }

void Track::SetElement(const size_t idx, const TrackElement& element) {
  elements_.at(idx) = element;
}

void Track::AddElement(const TrackElement& element) {
  // element may refer into elements_, so preserve it across replacement.
  const TrackElement value = element;
  PrepareAppend(1);
  elements_.push_back(value);
}

void Track::AddElement(const image_t image_id, const point2D_t point2D_idx) {
  PrepareAppend(1);
  elements_.emplace_back(image_id, point2D_idx);
}

void Track::AddElements(const std::vector<TrackElement>& elements) {
  if (mapping_credit_ && &elements == &elements_) {
    const Track copy(*this);
    AddElements(copy.elements_);
    return;
  }
  PrepareAppend(elements.size());
  elements_.insert(elements_.end(), elements.begin(), elements.end());
}

void Track::DeleteElement(const size_t idx) {
  THROW_CHECK_LT(idx, elements_.size());
  elements_.erase(elements_.begin() + idx);
}

void Track::Reserve(const size_t num_elements) {
  if (!mapping_credit_) {
    elements_.reserve(num_elements);
  } else if (num_elements > elements_.capacity()) {
    ReplaceCapacity(num_elements);
  }
}

void Track::Compress() {
  if (!mapping_credit_) {
    elements_.shrink_to_fit();
  } else if (elements_.size() < elements_.capacity()) {
    ReplaceCapacity(elements_.size());
  }
}

bool Track::operator==(const Track& other) const {
  return elements_ == other.elements_;
}

bool Track::operator!=(const Track& other) const { return !(*this == other); }

}  // namespace colmap
