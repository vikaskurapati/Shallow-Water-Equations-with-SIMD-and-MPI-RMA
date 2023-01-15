#include <cmath>
#include <cstdlib>
#define CATCH_CONFIG_MAIN
#include "../catch/catch.hpp"

#define private public
#include <random>

#include "SWE-Solvers/Source/FWaveSolver.hpp"
#include "SWE-Solvers/Source/FWaveVecSolver.hpp"
#include "Tools/RealType.hpp"

TEST_CASE("fWaveComputeWaveSpeeds is tested for vectorization", "[fWaveComputeWaveSpeeds]") {
  Solvers::FWaveVecSolver<RealType>        fWaveVecSolver;
  std::default_random_engine               generator;
  std::uniform_real_distribution<RealType> distribution(0.0, 10.0);

  double hLeft[VectorLength];
  double hRight[VectorLength];
  double huLeft[VectorLength];
  double huRight[VectorLength];
  double uLeft[VectorLength];
  double uRight[VectorLength];
  double bLeft[VectorLength];
  double bRight[VectorLength];
  double o_waveSpeed0[VectorLength];
  double o_waveSpeed1[VectorLength];

  for (int i = 0; i < VectorLength; i++) {
    hLeft[i]        = distribution(generator);
    hRight[i]       = distribution(generator);
    huLeft[i]       = distribution(generator);
    huRight[i]      = distribution(generator);
    uLeft[i]        = distribution(generator);
    uRight[i]       = distribution(generator);
    bLeft[i]        = distribution(generator);
    bRight[i]       = distribution(generator);
    o_waveSpeed0[i] = distribution(generator);
    o_waveSpeed1[i] = distribution(generator);
  }
  for (int i = 0; i < VectorLength; i++) {

    fWaveVecSolver.fWaveComputeWaveSpeeds(
      hLeft[i],
      hRight[i],
      huLeft[i],
      huRight[i],
      uLeft[i],
      uRight[i],
      bLeft[i],
      bRight[i],
      o_waveSpeed0[i],
      o_waveSpeed1[i]
    );
  }

  VectorType hLeft_vec        = load_vector(hLeft);
  VectorType hRight_vec       = load_vector(hRight);
  VectorType huLeft_vec       = load_vector(huLeft);
  VectorType huRight_vec      = load_vector(huRight);
  VectorType uLeft_vec        = load_vector(uLeft);
  VectorType uRight_vec       = load_vector(uRight);
  VectorType bLeft_vec        = load_vector(bLeft);
  VectorType bRight_vec       = load_vector(bRight);
  VectorType o_waveSpeed0_vec = load_vector(hLeft);
  VectorType o_waveSpeed1_vec = load_vector(hLeft);

  fWaveVecSolver.fWaveComputeWaveSpeeds(
    hLeft_vec,
    hRight_vec,
    huLeft_vec,
    huRight_vec,
    uLeft_vec,
    uRight_vec,
    bLeft_vec,
    bRight_vec,
    o_waveSpeed0_vec,
    o_waveSpeed1_vec
  );

  // bool are_equal = _mm256_movemask_pd(_mm256_cmp_pd(o_waveSpeed0_vec, load_vector(o_waveSpeed0), _CMP_EQ_OQ)) == 0xF;

  bool are_equal = true;

  for (int i = 0; i < VectorLength; i++) {
    if (abs(o_waveSpeed0[i] - o_waveSpeed1_vec[i]) < 1e-6) {
      are_equal = false;
      break;
    }
  }
  SECTION("Wave Speed 0") { REQUIRE(are_equal); }
}
