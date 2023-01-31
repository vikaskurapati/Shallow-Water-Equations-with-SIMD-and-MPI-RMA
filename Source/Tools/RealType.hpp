#include <immintrin.h>
#pragma once

#ifdef ENABLE_SINGLE_PRECISION
using RealType                                   = float;
#define MY_MPI_FLOAT MPI_FLOAT
#else
using RealType                                   = double;
#define MY_MPI_FLOAT MPI_DOUBLE
#endif
// Datatype for the type of data stored in the structures
#ifdef ENABLE_VECTORIZATION
#ifdef ENABLE_SINGLE_PRECISION
using RealType                                   = float;
using VectorType                                 = __m256;
constexpr int                       VectorLength = 8;
typedef decltype(_mm256_setzero_ps) set_vec_zero;
constexpr auto                      load_vector        = _mm256_loadu_ps;
constexpr auto set_vector_zero    = _mm256_setzero_ps;
constexpr auto                      square_root_vector = _mm256_sqrt_ps;
constexpr auto                      sub_vector         = _mm256_sub_ps;
constexpr auto                      add_vector         = _mm256_add_ps;
constexpr auto                      mul_vector         = _mm256_mul_ps;
constexpr auto                      div_vector         = _mm256_div_ps;
constexpr auto                      set_vector         = _mm256_set1_ps;
constexpr auto                      min_vector         = _mm256_min_ps;
constexpr auto                      max_vector         = _mm256_max_ps;
constexpr auto                      compare_vector     = _mm256_cmp_ps;
constexpr auto                      bitwise_and        = _mm256_and_ps;
constexpr auto                      bitwise_or         = _mm256_or_ps;
constexpr auto                      blend_vector       = _mm256_blendv_ps;
constexpr auto                      store_vector       = _mm256_storeu_ps;
constexpr auto                      maximum_vector     = _mm256_max_ps;

#define MY_MPI_FLOAT MPI_FLOAT
#else
using RealType                    = double;
using VectorType                  = __m256d;
constexpr int  VectorLength       = 4;
constexpr auto set_vector_zero    = _mm256_setzero_pd;
constexpr auto load_vector        = _mm256_loadu_pd;
constexpr auto square_root_vector = _mm256_sqrt_pd;
constexpr auto sub_vector         = _mm256_sub_pd;
constexpr auto add_vector         = _mm256_add_pd;
constexpr auto mul_vector         = _mm256_mul_pd;
constexpr auto div_vector         = _mm256_div_pd;
constexpr auto set_vector         = _mm256_set1_pd;
constexpr auto min_vector         = _mm256_min_pd;
constexpr auto max_vector         = _mm256_max_pd;
constexpr auto compare_vector     = _mm256_cmp_pd;
constexpr auto bitwise_and        = _mm256_and_pd;
constexpr auto bitwise_or         = _mm256_or_pd;
constexpr auto blend_vector       = _mm256_blendv_pd;
constexpr auto store_vector       = _mm256_storeu_pd;
constexpr auto maximum_vector     = _mm256_max_pd;

#define MY_MPI_FLOAT MPI_DOUBLE
#endif
#endif