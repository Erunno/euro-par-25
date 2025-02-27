#ifndef ROW_MACRO_HPP
#define ROW_MACRO_HPP

#define POPCOUNT_64(x) __popcll(x)
#define GOL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb) (((((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1)))) == (static_cast<std::uint64_t>(0b1)))) | (POPCOUNT_64(((((lc & static_cast<std::uint64_t>(0b11)) << 7)) | (((cc & static_cast<std::uint64_t>(0b10)) << 5)) | (((rc & static_cast<std::uint64_t>(0b11)) << 3)) | (((lt & static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000000000000000000)) >> 2)) | (((ct & static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000000000000000000)) >> 1)) | (rt & static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000000000000000000))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000000000000000000)))) | (POPCOUNT_64(((((lc & static_cast<std::uint64_t>(0b1100000000000000000000000000000000000000000000000000000000000000)) >> 7)) | (((cc & static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000000000000000000000)) >> 5)) | (((rc & static_cast<std::uint64_t>(0b1100000000000000000000000000000000000000000000000000000000000000)) >> 3)) | (((lb & static_cast<std::uint64_t>(0b1)) << 2)) | (((cb & static_cast<std::uint64_t>(0b1)) << 1)) | (rb & static_cast<std::uint64_t>(0b1))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10)))) == (static_cast<std::uint64_t>(0b10)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111)))) << (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101)))) << (3)) | ((lc) & (static_cast<std::uint64_t>(0b111)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100)))) == (static_cast<std::uint64_t>(0b100)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110)))) << (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010)))) << (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000)))) == (static_cast<std::uint64_t>(0b1000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100)))) << (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100)))) << (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000)))) == (static_cast<std::uint64_t>(0b10000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000)))) << (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000)))) << (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000)))) == (static_cast<std::uint64_t>(0b100000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000)))) << (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000)))) << (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000)))) == (static_cast<std::uint64_t>(0b1000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100000)))) << (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100000)))) << (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000000)))) == (static_cast<std::uint64_t>(0b10000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000000)))) == (static_cast<std::uint64_t>(0b100000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000000)))) == (static_cast<std::uint64_t>(0b1000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000000000)))) == (static_cast<std::uint64_t>(0b10000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000000000)))) == (static_cast<std::uint64_t>(0b100000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000000000)))) == (static_cast<std::uint64_t>(0b1000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000000000000)))) == (static_cast<std::uint64_t>(0b10000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000000000000)))) == (static_cast<std::uint64_t>(0b100000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000000000000)))) == (static_cast<std::uint64_t>(0b1000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000000000000000)))) == (static_cast<std::uint64_t>(0b10000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000000000000000)))) == (static_cast<std::uint64_t>(0b100000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000000000000000)))) == (static_cast<std::uint64_t>(0b1000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000000000000000000)))) == (static_cast<std::uint64_t>(0b10000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000000000000000000)))) == (static_cast<std::uint64_t>(0b100000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000000000000000000)))) == (static_cast<std::uint64_t>(0b1000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000000000000000000000)))) == (static_cast<std::uint64_t>(0b10000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000000000000000000000)))) == (static_cast<std::uint64_t>(0b100000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000000000000000000000)))) == (static_cast<std::uint64_t>(0b1000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000000000000000000000000)))) == (static_cast<std::uint64_t>(0b10000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000000000000000000000000)))) == (static_cast<std::uint64_t>(0b100000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b1000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b10000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b100000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b1000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b10000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b100000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b1000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b10000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b100000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100000000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000000000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000000000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100000000000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000000000000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000000000000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100000000000000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000000000000000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000000000000000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100000000000000000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000000000000000000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000000000000000000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b10100000000000000000000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b11100000000000000000000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b1000000000000000000000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b101000000000000000000000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b111000000000000000000000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b10000000000000000000000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))|(((static_cast<std::uint64_t>(((((((((cc) & (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000000000000000000000)))) == (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000000000000000000000)))) | (POPCOUNT_64((((((rc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000000000000000000000000000000)))) >> (6)) | ((((cc) & (static_cast<std::uint64_t>(0b1010000000000000000000000000000000000000000000000000000000000000)))) >> (3)) | ((lc) & (static_cast<std::uint64_t>(0b1110000000000000000000000000000000000000000000000000000000000000)))))))) == (3)))) ? (static_cast<std::uint64_t>(0b100000000000000000000000000000000000000000000000000000000000000)) : (static_cast<std::uint64_t>(0b0))))))))

#endif // ROW_MACRO_HPP