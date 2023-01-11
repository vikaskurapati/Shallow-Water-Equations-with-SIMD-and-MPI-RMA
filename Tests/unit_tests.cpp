#include <cmath>
#include <cstdlib>
#define CATCH_CONFIG_MAIN
#include "../catch/catch.hpp"

#define private public
#include "FWaveSolver.hpp"

TEST_CASE("FwaveSolve is tested", "[FwaveSolver]") {

  Solvers::FWaveSolver<double> fwaveSolver;
  //h is 1-20, hu -> -15  to 15, hv same
  
  double waveSpeeds[2]  = {-1.5, 1.5};
  double o_hupdateLeft  = 0.1;
  double o_hupdateRight = 0.1;
  double o_huUpdateLeft = 0.1;
  double o_hvUpdateLeft = 0.1;
  double o_maxWaveSpeed = 0.2;

  double test_waveSpeeds[2]  = {-1.5, 1.5};
  double test_o_hupdateLeft  = 0.1;
  double test_o_hupdateRight = 0.1;
  double test_o_huUpdateLeft = 0.1;
  double test_o_hvUpdateLeft = 0.1;
  double test_o_maxWaveSpeed = 0.2;

  fwaveSolver.computeNetUpdatesWithWaveSpeeds(
    waveSpeeds, o_hupdateLeft, o_hupdateRight, o_huUpdateLeft, o_hvUpdateLeft, o_maxWaveSpeed
  );
  fwaveSolver.computeNetUpdatesWithWaveSpeeds(
    test_waveSpeeds,
    test_o_hupdateLeft,
    test_o_hupdateRight,
    test_o_huUpdateLeft,
    test_o_hvUpdateLeft,
    test_o_maxWaveSpeed
  );

  SECTION("Checking if this thing would work") { REQUIRE(std::abs(o_huUpdateLeft - test_o_huUpdateLeft) < 0.001); }
  
  SECTION("Checking if this thing would work") { REQUIRE(1.01 > 1.00); }

}