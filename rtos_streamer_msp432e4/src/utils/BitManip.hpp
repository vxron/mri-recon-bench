// ARM is LITTLE-ENDIAN, 32-bit words

class BitManipulator_C {
    // POS ARE FROM LSB (LSB=pos 0), 'start'is towards LSB, 'end' is towards MSB
public:
    void bit_set(uint32_t& word, const std::size_t pos_start, std::optional<std::size_t> pos_end = std::nullopt); // sets from pos_start to pos_end, inclusive
    void bit_clear(uint32_t& word, const std::size_t pos_start, std::optional<std::size_t> pos_end = std::nullopt); // clears from pos_start to pos_end, inclusive
    void bit_toggle(uint32_t& word, const std::size_t pos_start, std::optional<std::size_t> pos_end = std::nullopt);
    void bit_write_val(uint32_t& word, const uint32_t val_to_write, const std::size_t pos_start, std::optional<std::size_t> pos_end = std::nullopt);
    void bit_read_val(const uint32_t& word, uint32_t& dest, const std::size_t pos_start, std::optional<std::size_t> pos_end = std::nullopt);
    std::size_t count_num_bits_word(const uint32_t val);
private:
    // reusable mask
    uint32_t mask_ = 0x0;
};

void BitManipulator_C::bit_set(uint32_t& word, const std::size_t pos_start, std::optional<std::size_t> pos_end) {
    if(pos_end == std::nullopt){
        pos_end = pos_start;
    } else {
        std::size_t pos_end_val = pos_end.value();
    }
    mask_ = 1u;
    for(std::size_t i=pos_start;i<=pos_end_val;i++){
        word |= (mask_ << i);
    }
}

void BitManipulator_C::bit_clear(uint32_t& word, const std::size_t pos_start, std::optional<std::size_t> pos_end) {
    if(pos_end == std::nullopt){
        pos_end = pos_start;
    } else {
        std::size_t pos_end_val = pos_end.value();
    }
    mask_ = 1u;
    for(std::size_t i=pos_start;i<=pos_end_val;i++){
        word &= ~(mask_ << i); // first shift 1 where we want it, then invert so it becomes a 0 and everything else is a 1 ! 
    }
}

void BitManipulator_C::bit_toggle(uint32_t& word, const std::size_t pos_start, std::optional<std::size_t> pos_end) {
    if(pos_end == std::nullopt){
        pos_end = pos_start;
    } else {
        std::size_t pos_end_val = pos_end.value();
    }
    mask_ = 1u;
    for(std::size_t i=pos_start;i<=pos_end_val;i++){
        word ^= (mask_ << i); // toggle where 1, all else 0
    }
} 

void BitManipulator_C::bit_read_val(const uint32_t& word, uint32_t& dest, const std::size_t pos_start, std::optional<std::size_t> pos_end){
    if(pos_end == std::nullopt){
        pos_end = pos_start;
    }
    // init dest if not done so
    dest = 0x0; 
    mask_ = 1u;

    for(int p = pos_end; p>=pos_start; p--){
        // descending order (MSB -> LSB)
        if(word & (mask_<<p) != 0){ // then we must have a 1 at pos p
            dest = (dest<<1)+0b1; 
        } else {
            dest = (dest<<1); // making room for new dig 0
        }
    }
}

std::size_t BitManipulator_C::count_num_bits_word(const uint32_t val){
    // it is essentially the idx of the most sig dig 
    std::size_t WORD_LENGTH = 32;
    uint32_t tmp = 0;
    std::size_t numBits = 32;
    // start from top idx (MSB) until we find first 1 -> that is MSB pos
    for(int i=WORD_LENGTH-1;i>=0;i--){
        bit_read_val(val, tmp, i, i);
        if(tmp==1){
            break;
        } 
        numBits--;
    }
    return numBits;
}

void BitManipulator_C::bit_write_val(uint32_t& word, const uint32_t val_to_write, const std::size_t pos_start, std::optional<std::size_t> pos_end){
    if(pos_end == std::nullopt){
        // allow no pos-end, infer based on val_to_write, but make sure it doesn't surpass 32 bits from pos_start
        if(count_num_bits_word(val_to_write) > (32-pos_start)) {
            return; // (ideally convert to bool and return false, or some other form of error coding)
        }
        // otherwise lets infer pos_end based on val_to_write
        pos_end = (count_num_bits_word(val_to_write)) + pos_start - 1;
    }
    // check val to write is same size as specified pos_end - pos_start if given
    if((count_num_bits_word(val_to_write) != (pos_end-pos_start+1))){
        return;
    }

    // start w LSB of val to write
    std::size_t val_idx = 0;
    for(std::size_t i=pos_start;i<=pos_end;i++){
        // take LSB of val to write
        if ((val_to_write >> val_idx) & 1u != 0) {
            // then we must have a 1 at this val_idx
            bit_set(word, i);
        } else {
            // we must have a 0 at this val_idx
            bit_clear(word, i);
        }
        val_idx++;
    }
}

class ByteWriter_C {

}

class ByteReader_C {

}