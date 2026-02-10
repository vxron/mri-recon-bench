#include <array>
#include <semaphore>

// NUM_SLOTS = length of buffer (e.g. 64)
template <typename T, std::size_t NUM_SLOTS>

class RingBuffer_C<T> {
public:
    explicit RingBuffer_C();
    bool close_RB();
    bool push(T& item);
    bool pop(T* dest);

    bool isEmpty() {
        return (curr_count_==0);
    }
    bool isFull() {
        return (curr_count_==NUM_SLOTS);
    }
    bool isClosed() {
        return (isClosed_ == true);
    }

private:
    std::size_t tail_idx_=0;
    std::size_t head_idx_=0;
    std::size_t curr_count_=0;

    std::lock_guard<std::mutex> arr_mtx_;
    std::array<T, NUM_SLOTS> buf_arr_;

    // 2 semaphores:
    // 1) for empty buf (consumer waits)
    std::counting_semaphore<NUM_SLOTS> sem_data_items_available(0); // start at 0, go to max count
    // 2) for full buf (producer waits)
    std::counting_semaphore<NUM_SLOTS> sem_buf_slots_available(0);

    bool isClosed_=true;
}

template<typename T>
RingBuffer_C<T>::RingBuffer_C() : isClosed_(false) {}

bool close_RB() {
    isClosed_ = true;
    return true;
}


bool push(T& item){

}


bool pop(T* dest);


