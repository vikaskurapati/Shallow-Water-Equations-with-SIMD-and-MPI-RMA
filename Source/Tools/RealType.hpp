#include <immintrin.h>
#include "immintrin.h"
#pragma once

// Datatype for the type of data stored in the structures
#ifdef ENABLE_SINGLE_PRECISION
using RealType = float;
using VectorType = __m256;
constexpr int VectorLength = 8;
typedef decltype(_mm256_setzero_ps) set_vec_zero;
constexpr auto load_vector = _mm256_load_ps;
constexpr auto square_root_vector = _mm256_sqrt_ps;
constexpr auto sub_vector = _mm256_sub_ps;
constexpr auto add_vector = _mm256_add_ps;
constexpr auto mul_vector = _mm256_mul_ps;
constexpr auto div_vector = _mm256_div_ps;
constexpr auto set_vector = _mm256_set1_ps;
#define MY_MPI_FLOAT MPI_FLOAT
#else
using RealType = double;
using VectorType = __m256d;
constexpr int VectorLength = 4;
constexpr auto set_vector_zero = _mm256_setzero_pd;
constexpr auto load_vector = _mm256_load_pd;
constexpr auto square_root_vector = _mm256_sqrt_pd;
constexpr auto sub_vector = _mm256_sub_pd;
constexpr auto add_vector = _mm256_add_pd;
constexpr auto mul_vector = _mm256_mul_pd;
constexpr auto div_vector = _mm256_div_pd;
constexpr auto set_vector = _mm256_set1_pd;
#define MY_MPI_FLOAT MPI_DOUBLE
#endif
