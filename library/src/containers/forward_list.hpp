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

#ifndef LIBRARY_SRC_CONTAINERS_FORWARD_LIST_HPP_
#define LIBRARY_SRC_CONTAINERS_FORWARD_LIST_HPP_

#include <hip/hip_runtime.h>

#include <initializer_list>
#include <ostream>
#include <string>

#include "src/device_proxy.hpp"
#include "src/memory/hip_allocator.hpp"
#include "src/sync/abql_block_mutex.hpp"

namespace rocshmem {

/*****************************************************************************
 ******************************* FORWARD LIST ********************************
 *****************************************************************************/

template <typename TYPE, typename ALLOC = HIPAllocator>
class ForwardList {
  friend class ForwardListTestFixture;

  struct Node {
    TYPE data;
    Node* next;
  };

  template <bool CONST>
  class Iterator;

 public:
  using iterator = Iterator<false>;
  using const_iterator = Iterator<true>;

  /**
   * @brief Default constructor
   */
  ForwardList() = default;

  explicit ForwardList(const ALLOC& alloc);

  /**
   * @brief Fill constructor
   */
  explicit ForwardList(size_t n, const ALLOC& alloc = ALLOC());

  explicit ForwardList(size_t n, const TYPE& val, const ALLOC& alloc = ALLOC());

  /**
   * @brief Range constructor
   */
  template <class InputIterator>
  ForwardList(InputIterator first, InputIterator last,
              const ALLOC& alloc = ALLOC());

  /**
   * @brief Copy constructor
   */
  ForwardList(const ForwardList& fwdlst);

  ForwardList(const ForwardList& fwdlst, const ALLOC& alloc);

  /**
   * @brief Move constructor
   */
  ForwardList(ForwardList&& fwdlst);

  ForwardList(ForwardList&& fwdlst, const ALLOC& alloc);

  /**
   * @brief Initializer list constructor
   */
  ForwardList(std::initializer_list<TYPE> il, const ALLOC& alloc = ALLOC());

  /**
   * @brief Destructor
   */
  ~ForwardList();

  /**
   * @brief Copy assignment operator
   */
  ForwardList& operator=(const ForwardList& fwdlst);

  /**
   * @brief Move assignment operator
   */
  ForwardList& operator=(ForwardList&& fwdlst);

  /**
   * @brief Initializer list assignment
   */
  ForwardList& operator=(std::initializer_list<TYPE> il);

  /**
   * @brief Returns iterator pointing to position before first element.
   *
   * The iterator returned shall not be dereferenced:
   * It is meant to be used as an argument for member functions
   * emplace_after, insert_after, erase_after or splice_after, to specify
   * the beginning of the sequence as the location where the action is
   * performed.
   */
  iterator before_begin() noexcept;

  /**
   * @brief Returns iterator pointing to first element in ForwardList.
   *
   * Notice that, unlike member ForwardList::front, which returns a
   * reference to the first element, this function returns a forward iterator
   * pointing to it.
   *
   * If the container is empty, the returned iterator value shall not be
   * dereferenced.
   */
  iterator begin() noexcept;

  const_iterator begin() const noexcept;

  /**
   * @brief Returns an iterator to past-the-end element in ForwardList.
   */
  iterator end() noexcept;

  const_iterator end() const noexcept;

  /**
   * @brief Returns const_iterator pointing to position before first element.
   */
  const_iterator cbefore_begin() const noexcept;

  /**
   * @brief Returns const_iterator pointing to first element.
   */
  const_iterator cbegin() const noexcept;

  /**
   * @brief Returns const_iterator pointing to past-the-end element.
   */
  const_iterator cend() const noexcept;

  /**
   * @brief Returns bool indicating if ForwardList container is empty.
   */
  bool empty() const noexcept;

  /**
   * @brief Returns maximum number of elements that ForwardList can hold.
   */
  size_t max_size() const noexcept;

  /**
   * @brief Returns reference to first element in the ForwardList.
   */
  ForwardList& front();

  const ForwardList& front() const;

  /**
   * @brief Assigns new values, replacing current contents, and modify size.
   *
   * In range version, the new contents are elements constructed from each
   * of the elements in the range between first and last, in the same order.
   */
  template <class InputIterator>
  void assign(InputIterator first, InputIterator last);

  /**
   * @brief Assigns new values, replacing current contents, and modify size.
   *
   * In fill version, the new contents are n elements, each initialized to
   * a copy of val.
   */
  void assign(size_t n, const TYPE& val);

  /**
   * @brief Assigns new values, replacing current contents, and modify size.
   *
   * In initializer list version, the new contents are copies of the values
   * passed as initializer list, in the same order.
   */
  void assign(std::initializer_list<TYPE> il);

  /**
   * @brief Inserts new element at beginning of the ForwardList.
   *
   * The element goes into the container right before its current first
   * element. This new element is constructed in place using args as the
   * arguments for its construction.
   */
  template <class... Args>
  void emplace_front(Args&&... args);

  /**
   * @brief Inserts new element at the beginning of the ForwardList.
   *
   * The element goes into the container right before its current first
   * element. The content of val is copied (or moved) to the inserted
   * element.
   */
  void push_front(const TYPE& val);

  void push_front(TYPE&& val);

  /**
   * @brief Removes first element in ForwardList, reducing its size by one.
   */
  void pop_front();

  /**
   * @brief Inserts a new element after element at position.
   *
   * This new element is constructed in place using args as the arguments
   * for its construction.
   */
  template <class... Args>
  iterator emplace_after(const_iterator position, Args&&... args);

  /**
   * @brief Inserts new elements after the element at position.
   */
  iterator insert_after(const_iterator position, const TYPE& val);

  iterator insert_after(const_iterator position, TYPE&& val);

  iterator insert_after(const_iterator position, size_t n, const TYPE& val);

  template <class InputIterator>
  iterator insert_after(const_iterator position, InputIterator first,
                        InputIterator last);

  iterator insert_after(const_iterator position,
                        std::initializer_list<TYPE> il);

  /**
   * @brief Removes either a single element or a range of elements.
   */
  iterator erase_after(const_iterator position);

  iterator erase_after(const_iterator position, const_iterator last);

  /**
   * @brief Exchanges content by the content of fwdlst.
   *
   * Sizes may differ.
   */
  void swap(ForwardList& fwdlst);

  /**
   * @brief Resizes the container to contain n elements.
   */
  void resize(size_t n);

  void resize(size_t n, const TYPE& val);

  /**
   * @brief Removes all elements, all leaves container with size 0.
   */
  void clear() noexcept;

  /**
   * @brief Transfers all elements of fwdlist into container.
   */
  void splice_after(const_iterator position, ForwardList& fwdlst);  // NOLINT

  void splice_after(const_iterator position, ForwardList&& fwdlst);

  /**
   * @brief Transfers only elements pointed by from fwdlist into container.
   */
  void splice_after(const_iterator position, ForwardList& fwdlst,  // NOLINT
                    const_iterator i);

  void splice_after(const_iterator position, ForwardList&& fwdlst,
                    const_iterator i);

  /**
   * @brief Transfers the range (first,last) from fwdlist into container.
   */
  void splice_after(const_iterator position, ForwardList& fwdlst,  // NOLINT
                    const_iterator first, const_iterator last);

  void splice_after(const_iterator position, ForwardList&& fwdlst,  // NOLINT
                    const_iterator first, const_iterator last);

  /**
   * @brief Removes all the elements that compare equal to val.
   */
  void remove(const TYPE& val);

  /**
   * @brief Removes all elements for which Predicate pred returns true.
   */
  template <class Predicate>
  void remove_if(Predicate pred);

  /**
   * @brief Removes all but first element from consecutive group equal elems.
   */
  void unique();

  /**
   * @brief Takes comparison func that determines "uniqueness" of elem.
   */
  template <class BinaryPredicate>
  void unique(BinaryPredicate binary_pred);

  /**
   * @brief Merges ForwardList by transferring all elements.
   *
   * Both containers shall already be ordered before calling merge.
   */
  void merge(ForwardList& fwdlst);  // NOLINT

  void merge(ForwardList&& fwdlst);  // NOLINT

  /**
   * @brief Same as merge, but take specific predicate to perform comparison.
   */
  template <class Compare>
  void merge(ForwardList& fwdlst, Compare comp);  // NOLINT

  template <class Compare>
  void merge(ForwardList&& fwdlst, Compare comp);  // NOLINT

  /**
   * @brief Sorts elements in ForwardList, altering position in container.
   */
  void sort();

  template <class Compare>
  void sort(Compare comp);  // NOLINT

  /**
   * @brief Reverses order of elements in the ForwardList container.
   */
  void reverse() noexcept;

  /**
   * @brief Returns a copy of the allocator object associated with container.
   */
  ALLOC
  get_allocator() const noexcept;

 private:
  /**
   * @brief Internal memory allocator used to create list nodes.
   */
  MemoryAllocator allocator_{};

  /**
   * @brief First element in the list.
   */
  Node* head_{nullptr};

  /**
   * @brief Last element in the list.
   */
  Node* tail_{nullptr};

  /**
   * @brief Size of the list.
   */
  size_t size_{0};
};

/*****************************************************************************
 ********************************* ITERATOR **********************************
 *****************************************************************************/

template <typename TYPE, typename ALLOC>
template <bool CONST>
class ForwardList<TYPE, ALLOC>::Iterator {
  using NodeT = typename ForwardList<TYPE, ALLOC>::Node;

 public:
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = TYPE;
  using pointer = TYPE*;
  using reference = TYPE&;

  Iterator(NodeT* ptr);

  Iterator operator++();

  Iterator operator++(int);

  template <bool Q = CONST>
  typename std::enable_if<Q, const TYPE&>::type operator*();

  template <bool Q = CONST>
  typename std::enable_if<!Q, TYPE&>::type operator*();

  pointer operator->();

  template <typename ITER_TYPE>
  friend bool operator==(ITER_TYPE& a, ITER_TYPE& b);  // NOLINT

  template <typename ITER_TYPE>
  friend bool operator==(ITER_TYPE& a, ITER_TYPE b);  // NOLINT

  template <typename ITER_TYPE>
  friend bool operator!=(ITER_TYPE& a, ITER_TYPE& b);  // NOLINT

  template <typename ITER_TYPE>
  friend bool operator!=(ITER_TYPE& a, ITER_TYPE b);  // NOLINT

 private:
  NodeT* node_ptr_;
};

/*****************************************************************************
 ******************************* COMPARATORS *********************************
 *****************************************************************************/

template <class TYPE, class ALLOC>
bool operator==(const ForwardList<TYPE, ALLOC>& lhs,
                const ForwardList<TYPE, ALLOC>& rhs);

template <class TYPE, class ALLOC>
bool operator==(const ForwardList<TYPE, ALLOC>& lhs, const std::string rhs);

template <class TYPE, class ALLOC>
bool operator==(const std::string lhs, const ForwardList<TYPE, ALLOC>& rhs);

template <class TYPE, class ALLOC>
bool operator!=(const ForwardList<TYPE, ALLOC>& lhs,
                const ForwardList<TYPE, ALLOC>& rhs);

template <class TYPE, class ALLOC>
bool operator<(const ForwardList<TYPE, ALLOC>& lhs,
               const ForwardList<TYPE, ALLOC>& rhs);

template <class TYPE, class ALLOC>
bool operator<=(const ForwardList<TYPE, ALLOC>& lhs,
                const ForwardList<TYPE, ALLOC>& rhs);

template <class TYPE, class ALLOC>
bool operator>(const ForwardList<TYPE, ALLOC>& lhs,
               const ForwardList<TYPE, ALLOC>& rhs);

template <class TYPE, class ALLOC>
bool operator>=(const ForwardList<TYPE, ALLOC>& lhs,
                const ForwardList<TYPE, ALLOC>& rhs);

template <class TYPE, class ALLOC>
void swap(ForwardList<TYPE, ALLOC>& x, ForwardList<TYPE, ALLOC>& y);  // NOLINT

/*****************************************************************************
 ******************************** STRINGIFIERS *******************************
 *****************************************************************************/

template <class TYPE>
std::ostream& operator<<(std::ostream& os, ForwardList<TYPE> const& list);

template <class TYPE, class ALLOC>
std::string to_string(const ForwardList<TYPE, ALLOC>& list);

/*****************************************************************************
 *********************************** PROXY ***********************************
 *****************************************************************************/

template <typename ALLOC, typename TYPE>
class ForwardListProxy {
  using ProxyT = DeviceProxy<ALLOC, ForwardList<TYPE, ALLOC>, 1>;

 public:
  __host__ __device__ ForwardList<TYPE, ALLOC>* get() { return proxy_.get(); }

 private:
  ProxyT proxy_{};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONTAINERS_FORWARD_LIST_HPP_
