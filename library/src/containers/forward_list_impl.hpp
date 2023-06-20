/******************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_CONTAINERS_FORWARD_LIST_IMPL_HPP_
#define LIBRARY_SRC_CONTAINERS_FORWARD_LIST_IMPL_HPP_

#include <algorithm>
#include <cassert>
#include <string>

#include "src/containers/forward_list.hpp"

namespace rocshmem {

/*****************************************************************************
 ******************************* FORWARD LIST ********************************
 *****************************************************************************/

template <typename TYPE, typename ALLOC>
ForwardList<TYPE, ALLOC>::~ForwardList() {
  while (head_ != nullptr) {
    Node* temp{head_};
    head_ = head_->next;
    allocator_.deallocate(temp);
  }
  tail_ = nullptr;
}

template <typename TYPE, typename ALLOC>
ForwardList<TYPE, ALLOC>::ForwardList(const ForwardList& fwdlst) {}

template <typename TYPE, typename ALLOC>
ForwardList<TYPE, ALLOC>::ForwardList(const ForwardList& fwdlst,
                                      const ALLOC& alloc) {}

template <typename TYPE, typename ALLOC>
ForwardList<TYPE, ALLOC>::ForwardList(std::initializer_list<TYPE> il,
                                      const ALLOC& allocator) {
  allocator_ = allocator;
  assign(il);
}

template <typename TYPE, typename ALLOC>
typename ForwardList<TYPE, ALLOC>::iterator
ForwardList<TYPE, ALLOC>::begin() noexcept {
  iterator i(head_);
  return i;
}

template <typename TYPE, typename ALLOC>
typename ForwardList<TYPE, ALLOC>::const_iterator
ForwardList<TYPE, ALLOC>::begin() const noexcept {
  const_iterator i(head_);
  return i;
}

template <typename TYPE, typename ALLOC>
typename ForwardList<TYPE, ALLOC>::iterator
ForwardList<TYPE, ALLOC>::end() noexcept {
  iterator i(tail_);
  return i;
}

template <typename TYPE, typename ALLOC>
typename ForwardList<TYPE, ALLOC>::const_iterator
ForwardList<TYPE, ALLOC>::end() const noexcept {
  const_iterator i(tail_);
  return i;
}

template <typename TYPE, typename ALLOC>
void ForwardList<TYPE, ALLOC>::assign(std::initializer_list<TYPE> il) {
  resize(il.size());
  std::copy_n(il.begin(), il.size(), begin());
}

template <typename TYPE, typename ALLOC>
void ForwardList<TYPE, ALLOC>::resize(size_t n) {}

template <typename TYPE, typename ALLOC>
void ForwardList<TYPE, ALLOC>::resize(size_t n, const TYPE& val) {}

template <typename TYPE, typename ALLOC>
void ForwardList<TYPE, ALLOC>::clear() noexcept {}

/*****************************************************************************
 ********************************* ITERATOR **********************************
 *****************************************************************************/

template <typename TYPE, typename ALLOC>
template <bool CONST>
ForwardList<TYPE, ALLOC>::Iterator<CONST>::Iterator(NodeT* ptr)
    : node_ptr_(ptr) {}

template <typename TYPE, typename ALLOC>
template <bool CONST>
typename ForwardList<TYPE, ALLOC>::template Iterator<CONST>
ForwardList<TYPE, ALLOC>::Iterator<CONST>::operator++() {
  if (node_ptr_) {
    node_ptr_ = node_ptr_->next;
  }
  return *this;
}

template <typename TYPE, typename ALLOC>
template <bool CONST>
typename ForwardList<TYPE, ALLOC>::template Iterator<CONST>
ForwardList<TYPE, ALLOC>::Iterator<CONST>::operator++(int) {
  Iterator iterator = *this;
  ++*this;
  return iterator;
}

template <typename TYPE, typename ALLOC>
template <bool CONST>
template <bool Q>
typename std::enable_if<!Q, TYPE&>::type
ForwardList<TYPE, ALLOC>::Iterator<CONST>::operator*() {
  return node_ptr_->data;
}

template <typename TYPE, typename ALLOC>
template <bool CONST>
template <bool Q>
typename std::enable_if<Q, const TYPE&>::type
ForwardList<TYPE, ALLOC>::Iterator<CONST>::operator*() {
  return node_ptr_->data;
}

template <typename TYPE, typename ALLOC>
template <bool CONST>
TYPE* ForwardList<TYPE, ALLOC>::Iterator<CONST>::operator->() {
  return &node_ptr_->data;
}

template <typename ITER_TYPE>
bool operator==(ITER_TYPE& a, ITER_TYPE& b) {  // NOLINT
  return a.node_ptr_ == b.node_ptr_;
}

template <typename ITER_TYPE>
bool operator==(ITER_TYPE& a, ITER_TYPE b) {  // NOLINT
  return a.node_ptr_ == b.node_ptr_;
}

template <typename ITER_TYPE>
bool operator!=(ITER_TYPE& a, ITER_TYPE& b) {  // NOLINT
  return a.node_ptr_ != b.node_ptr_;
}

template <typename ITER_TYPE>
bool operator!=(ITER_TYPE& a, ITER_TYPE b) {  // NOLINT
  return a.node_ptr_ != b.node_ptr_;
}

/*****************************************************************************
 ******************************* COMPARATORS *********************************
 *****************************************************************************/

template <typename TYPE, typename ALLOC>
bool operator==(const ForwardList<TYPE, ALLOC>& lhs,
                const ForwardList<TYPE, ALLOC>& rhs) {
  return false;
}

template <typename TYPE, typename ALLOC>
bool operator==(const ForwardList<TYPE, ALLOC>& lhs, const std::string rhs) {
  return false;
}

template <typename TYPE, typename ALLOC>
bool operator==(const std::string lhs, const ForwardList<TYPE, ALLOC>& rhs) {
  return false;
}

template <typename TYPE, typename ALLOC>
bool operator!=(const ForwardList<TYPE, ALLOC>& lhs,
                const ForwardList<TYPE, ALLOC>& rhs) {
  return false;
}

template <typename TYPE, typename ALLOC>
bool operator!=(const ForwardList<TYPE, ALLOC>& lhs, const std::string rhs) {
  return false;
}

template <typename TYPE, typename ALLOC>
bool operator!=(const std::string lhs, const ForwardList<TYPE, ALLOC>& rhs) {
  return false;
}

/*****************************************************************************
 ******************************** STRINGIFIERS *******************************
 *****************************************************************************/

template <typename TYPE>
std::ostream& operator<<(std::ostream& os, ForwardList<TYPE> const& list) {
  using CItr = typename ForwardList<TYPE>::const_iterator;
  for (CItr ci = list.begin(); ci != list.end(); ++ci) {
    os << *ci << " ";
  }
  return os;
}

template <typename TYPE, typename ALLOC>
std::string to_string(const ForwardList<TYPE, ALLOC>& list) {
  std::stringstream ss;
  ss << list;
  return ss.str();
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONTAINERS_FORWARD_LIST_IMPL_HPP_
