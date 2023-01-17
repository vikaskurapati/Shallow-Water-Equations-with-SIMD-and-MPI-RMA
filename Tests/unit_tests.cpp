#include <cmath>
#include <cstdlib>
#define CATCH_CONFIG_MAIN
#include "../catch/catch.hpp"

#define private public
#include <random>

#include "SWE-Solvers/Source/FWaveSolver.hpp"
#include "SWE-Solvers/Source/FWaveVecSolver.hpp"
#include "Tools/RealType.hpp"
#include <iostream>


TEST_CASE("fWaveComputeWaveDecomposition is tested for vectorization", "[fWaveComputeWaveDecomposition]") {
  Solvers::FWaveVecSolver<RealType>        fWaveVecSolver;
  std::default_random_engine               generator;
  std::uniform_real_distribution<RealType> distribution(0.0, 10.0);

  RealType hLeft[VectorLength];
  RealType hRight[VectorLength];
  RealType huLeft[VectorLength];
  RealType huRight[VectorLength];
  RealType uLeft[VectorLength];
  RealType uRight[VectorLength];
  RealType bLeft[VectorLength];
  RealType bRight[VectorLength];
  RealType waveSpeed0[VectorLength];
  RealType waveSpeed1[VectorLength];
  RealType o_fWave0[VectorLength];
  RealType o_fWave1[VectorLength];

  for (int i = 0; i < VectorLength; i++) {
    hLeft[i]        = distribution(generator);
    hRight[i]       = distribution(generator);
    huLeft[i]       = distribution(generator);
    huRight[i]      = distribution(generator);
    uLeft[i]        = distribution(generator);
    uRight[i]       = distribution(generator);
    bLeft[i]        = distribution(generator);
    bRight[i]       = distribution(generator);
    waveSpeed0[i] = distribution(generator);
    waveSpeed1[i] = distribution(generator);

  }
  for (int i = 0; i < VectorLength; i++) {

    fWaveVecSolver.fWaveComputeWaveDecomposition(
      hLeft[i],
      hRight[i],
      huLeft[i],
      huRight[i],
      uLeft[i],
      uRight[i],
      bLeft[i],
      bRight[i],
      waveSpeed0[i],
      waveSpeed1[i],
      o_fWave0[i],
      o_fWave1[i]
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
  VectorType waveSpeed0_vec = load_vector(waveSpeed0);
  VectorType waveSpeed1_vec = load_vector(waveSpeed1);
  VectorType o_fWave0_vec;
  VectorType o_fWave1_vec;

  fWaveVecSolver.fWaveComputeWaveDecomposition(
    hLeft_vec,
    hRight_vec,
    huLeft_vec,
    huRight_vec,
    uLeft_vec,
    uRight_vec,
    bLeft_vec,
    bRight_vec,
    waveSpeed0_vec,
    waveSpeed1_vec,
    o_fWave0_vec,
    o_fWave1_vec
  );

  bool are_equal = true;

  for (int i = 0; i < VectorLength; i++) {
    if (abs(o_fWave0[i] - o_fWave0_vec[i]) > 1e-6) {
      are_equal = false;
      break;
    }
  }
  SECTION("fWave Speed 0") { REQUIRE(are_equal); }
  
  are_equal = true;

  for (int i = 0; i < VectorLength; i++) {
    if (abs(o_fWave1[i] - o_fWave1_vec[i]) > 1e-6) {
      are_equal = false;
      break;
    }
  }
  SECTION("fWave Speed 1") { REQUIRE(are_equal); }
}

TEST_CASE("fWaveComputeWaveSpeeds is tested for vectorization", "[fWaveComputeWaveSpeeds]") {
  Solvers::FWaveVecSolver<RealType>        fWaveVecSolver;
  std::default_random_engine               generator;
  std::uniform_real_distribution<RealType> distribution(0.0, 10.0);

  RealType hLeft[VectorLength];
  RealType hRight[VectorLength];
  RealType huLeft[VectorLength];
  RealType huRight[VectorLength];
  RealType uLeft[VectorLength];
  RealType uRight[VectorLength];
  RealType bLeft[VectorLength];
  RealType bRight[VectorLength];
  RealType o_waveSpeed0[VectorLength];
  RealType o_waveSpeed1[VectorLength];

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

  bool are_equal = true;

  for (int i = 0; i < VectorLength; i++) {
    if (abs(o_waveSpeed0[i] - o_waveSpeed0_vec[i]) > 1e-6) {
      are_equal = false;
      break;
    }
  }
  SECTION("Wave Speed 0") { REQUIRE(are_equal); }
  
  are_equal = true;

  for (int i = 0; i < VectorLength; i++) {
    if (abs(o_waveSpeed1[i] - o_waveSpeed1_vec[i]) > 1e-6) {
      are_equal = false;
      break;
    }
  }
  SECTION("Wave Speed 1") { REQUIRE(are_equal); }
}
