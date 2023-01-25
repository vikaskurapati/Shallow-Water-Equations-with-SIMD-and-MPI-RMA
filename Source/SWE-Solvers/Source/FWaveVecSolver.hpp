/**
 * FWaveVecSolver.hpp
 *
 ****
 **** This is a vectorizable C++ implementation of the F-Wave solver (FWaveSolver.hpp).
 ****
 *
 * Created on: Nov 13, 2012
 * Last Update: Dec 28, 2013
 *
 ****
 *
 *  Author: Sebastian Rettenberger
 *    Homepage: http://www5.in.tum.de/wiki/index.php/Sebastian_Rettenberger,_M.Sc.
 *    E-Mail: rettenbs AT in.tum.de
 *  Some optimzations: Michael Bader
 *    Homepage: http://www5.in.tum.de/wiki/index.php/Michael_Bader
 *    E-Mail: bader AT in.tum.de
 *
 ****
 *
 * (Main) Literature:
 *
 *   @article{bale2002wave,
 *            title={A wave propagation method for conservation laws and balance laws with spatially varying flux
 *functions}, author={Bale, D.S. and LeVeque, R.J. and Mitran, S. and Rossmanith, J.A.}, journal={SIAM Journal on
 *Scientific Computing}, volume={24}, number={3}, pages={955--978}, year={2002}}
 *
 *   @book{leveque2002finite,
 *         Author = {LeVeque, R. J.},
 *         Publisher = {Cambridge University Press},
 *         Title = {Finite Volume Methods for Hyperbolic Problems},
 *         Volume = {31},
 *         Year = {2002}}
 *
 *   @webpage{levequeclawpack,
 *            Author = {LeVeque, R. J.},
 *            Lastchecked = {January, 05, 2011},
 *            Title = {Clawpack Sofware},
 *            Url = {https://github.com/clawpack/clawpack-4.x/blob/master/geoclaw/2d/lib}}
 *
 ****
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <immintrin.h>
#include <iostream>

#include "Tools/RealType.hpp"

// void print_mm256d(__m256d vec) {
//   double* v = (double*) &vec;
//   printf("%f %f %f %f\n", v[0], v[1], v[2], v[3]);
// }

namespace Solvers {

  template <class T>
  class FWaveVecSolver {
  private:
    const T dryTol_;
    const T zeroTol_;
    const T halfGravity_;
    const T sqrtGravity_;

  public:
    /**
     * FWaveVec Constructor, takes three problem parameters
     * @param dryTol "dry tolerance": if the water height falls below dryTol, wall boundary conditions are applied
     * (default value is 100)
     * @param gravity takes the value of the gravity constant (default value is 9.81 m/s^2)
     * @param zeroTol computed f-waves with an absolute value < zeroTol are treated as static waves (default value is
     * 10^{-7})
     */
    FWaveVecSolver(T dryTol = T(1.0), T gravity = T(9.81), T zeroTol = T(0.0000001)):
      dryTol_(dryTol),
      zeroTol_(zeroTol),
      halfGravity_(T(0.5) * gravity),
      sqrtGravity_(std::sqrt(gravity)) {}

    /**
     * Takes the water height, discharge and bathymatry in the left and right cell
     * and computes net updates (left and right going waves) according to the f-wave approach.
     * It also returns the maximum wave speed.
     */
#ifdef ENABLE_VECTORIZATION
#pragma omp declare simd
#endif
    void computeNetUpdates(
      T  hLeft,
      T  hRight,
      T  huLeft,
      T  huRight,
      T  bLeft,
      T  bRight,
      T& o_hUpdateLeft,
      T& o_hUpdateRight,
      T& o_huUpdateLeft,
      T& o_huUpdateRight,
      T& o_maxWaveSpeed
    ) const {
      if (hLeft >= dryTol_) {
        if (hRight < dryTol_) {
          // Dry/Wet case
          // Set values according to wall boundary condition
          hRight  = hLeft;
          huRight = -huLeft;
          bRight  = bLeft;
        }
      } else if (hRight >= dryTol_) {
        // Wet/Dry case
        // Set values according to wall boundary condition
        hLeft  = hRight;
        huLeft = -huRight;
        bLeft  = bRight;
      } else {
        // Dry/Dry case
        // Set dummy values such that the result is zero
        hLeft   = dryTol_;
        huLeft  = T(0.0);
        bLeft   = T(0.0);
        hRight  = dryTol_;
        huRight = T(0.0);
        bRight  = T(0.0);
      }

      // Velocity on the left side of the edge
      T uLeft = huLeft / hLeft; // 1 FLOP (div)
      // Velocity on the right side of the edge
      T uRight = huRight / hRight; // 1 FLOP (div)

      /// Wave spweeds of the f-waves
      T waveSpeeds0 = T(0.0);
      T waveSpeeds1 = T(0.0);

      // std::cout
      //   << hLeft << " ; " << hRight << " ; " << huLeft << " ; " << huRight << " ; " << uLeft << " ; " << uRight << "
      //   ; "
      //   << bLeft << " ; " << bRight << std::endl;

      fWaveComputeWaveSpeeds(
        hLeft, hRight, huLeft, huRight, uLeft, uRight, bLeft, bRight, waveSpeeds0, waveSpeeds1
      ); // 20 FLOPs (incl. 3 sqrt, 1 div, 2 min/max)

      // Variables to store the two f-waves
      T fWaves0 = T(0.0);
      T fWaves1 = T(0.0);

      // Compute the decomposition into f-waves
      fWaveComputeWaveDecomposition(
        hLeft, hRight, huLeft, huRight, uLeft, uRight, bLeft, bRight, waveSpeeds0, waveSpeeds1, fWaves0, fWaves1
      ); // 23 FLOPs (incl. 1 div)

      // Compute the net-updates
      o_hUpdateLeft   = T(0.0);
      o_hUpdateRight  = T(0.0);
      o_huUpdateLeft  = T(0.0);
      o_huUpdateRight = T(0.0);

      // 1st wave family
      if (waveSpeeds0 < -zeroTol_) { // Left going
        o_hUpdateLeft += fWaves0;
        o_huUpdateLeft += fWaves0 * waveSpeeds0; // 3 FLOPs (assume left going wave ...)
      } else if (waveSpeeds0 > zeroTol_) {       // Right going
        o_hUpdateRight += fWaves0;
        o_huUpdateRight += fWaves0 * waveSpeeds0;
      } else { // Split waves, if waveSpeeds0 close to 0
        o_hUpdateLeft += T(0.5) * fWaves0;
        o_huUpdateLeft += T(0.5) * fWaves0 * waveSpeeds0;
        o_hUpdateRight += T(0.5) * fWaves0;
        o_huUpdateRight += T(0.5) * fWaves0 * waveSpeeds0;
      }

      // 2nd wave family
      if (waveSpeeds1 > zeroTol_) { // Right going
        o_hUpdateRight += fWaves1;
        o_huUpdateRight += fWaves1 * waveSpeeds1; // 3 FLOPs (assume right going wave ...)
      } else if (waveSpeeds1 < -zeroTol_) {       // Left going
        o_hUpdateLeft += fWaves1;
        o_huUpdateLeft += fWaves1 * waveSpeeds1;
      } else { // Split waves
        o_hUpdateLeft += T(0.5) * fWaves1;
        o_huUpdateLeft += T(0.5) * fWaves1 * waveSpeeds1;
        o_hUpdateRight += T(0.5) * fWaves1;
        o_huUpdateRight += T(0.5) * fWaves1 * waveSpeeds1;
      }

      // Compute maximum wave speed (-> CFL-condition)
      o_maxWaveSpeed = std::max(std::abs(waveSpeeds0), std::abs(waveSpeeds1)); // 3 FLOPs (2 abs, 1 max)

      // ========================
      // 54 FLOPs (3 sqrt, 4 div, 2 abs, 3 min/max)
    }

    void computeNetUpdates_SIMD(
      // VectorType hLeft, VectorType hRight, VectorType huLeft, VectorType huRight, VectorType bLeft, VectorType bRight
      // VectorType& o_hUpdateLeft
      // VectorType& o_hUpdateRight,
      // VectorType& o_huUpdateLeft,
      // VectorType& o_huUpdateRight,
      // VectorType& o_maxWaveSpeed
      const RealType* const i_hLeft,
      const RealType* const i_hRight,
      const RealType* const i_huLeft,
      const RealType* const i_huRight,
      const RealType* const i_bLeft,
      const RealType* const i_bRight,

      RealType* o_hUpdateLeft,
      RealType* o_hUpdateRight,
      RealType* o_huUpdateLeft,
      RealType* o_huUpdateRight,
      RealType& o_maxWaveSpeed
    ) {
      VectorType hLeft   = load_vector(i_hLeft);
      VectorType hRight  = load_vector(i_hRight);
      VectorType huLeft  = load_vector(i_huLeft);
      VectorType huRight = load_vector(i_huRight);
      VectorType bLeft   = load_vector(i_bLeft);
      VectorType bRight  = load_vector(i_bRight);
      VectorType dryTol_vec = set_vector(dryTol_);


      /// Unable to print stuff from here
      VectorType cmp1       = compare_vector(hLeft, dryTol_vec, _CMP_GE_OQ);
      VectorType cmp2       = compare_vector(hRight, dryTol_vec, _CMP_LT_OQ);

      // std::cout << "Mom come pick me up. I am scared. I am at add, line 294" << std::endl;

      // double* v = (double*)&cmp1;
      // printf("%f %f %f %f\n", v[0], v[1], v[2], v[3]);
      VectorType mask1      = bitwise_and(cmp1, cmp2);

      VectorType cmp3  = compare_vector(hRight, dryTol_vec, _CMP_GE_OQ);
      VectorType cmp4  = compare_vector(hLeft, dryTol_vec, _CMP_LT_OQ);
      VectorType mask2 = bitwise_and(cmp3, cmp4);

      VectorType cmp5  = compare_vector(hRight, dryTol_vec, _CMP_LT_OQ);
      VectorType mask3 = bitwise_and(cmp5, cmp4);

      hRight  = blend_vector(hRight, hLeft, mask1);
      huRight = blend_vector(huRight, mul_vector(huLeft, set_vector(-1.0)), mask1);
      bRight  = blend_vector(bRight, bLeft, mask1);

      hLeft  = blend_vector(hLeft, hRight, mask2);
      huLeft = blend_vector(huLeft, mul_vector(huRight, set_vector(-1.0)), mask2);
      bLeft  = blend_vector(bLeft, bRight, mask2);

      hLeft   = blend_vector(hLeft, dryTol_vec, mask3);
      huLeft  = blend_vector(huLeft, set_vector_zero(), mask3);
      bLeft   = blend_vector(bLeft, set_vector_zero(), mask3);
      hRight  = blend_vector(hRight, dryTol_vec, mask3);
      huRight = blend_vector(huRight, set_vector_zero(), mask3);
      bRight  = blend_vector(bRight, set_vector_zero(), mask3);

      VectorType uLeft  = div_vector(huLeft, hLeft);
      VectorType uRight = div_vector(huRight, hRight);

      VectorType waveSpeeds0 = set_vector_zero();
      VectorType waveSpeeds1 = set_vector_zero();

      fWaveComputeWaveSpeeds(hLeft, hRight, huLeft, huRight, uLeft, uRight, bLeft, bRight, waveSpeeds0, waveSpeeds1);

      VectorType fWaves0 = set_vector_zero();
      VectorType fWaves1 = set_vector_zero();

      fWaveComputeWaveDecomposition(
        hLeft, hRight, huLeft, huRight, uLeft, uRight, bLeft, bRight, waveSpeeds0, waveSpeeds1, fWaves0, fWaves1
      );

      VectorType o_hUpdateLeft_vec = set_vector_zero();

      // o_hUpdateLeft = set_vector_zero();

      VectorType o_hUpdateRight_vec  = set_vector_zero();
      VectorType o_huUpdateLeft_vec  = set_vector_zero();
      VectorType o_huUpdateRight_vec = set_vector_zero();

      VectorType zeroTol_vec = set_vector(zeroTol_);

      VectorType mask4 = compare_vector(waveSpeeds0, mul_vector(zeroTol_vec, set_vector(-1.0)), _CMP_LT_OQ);
      // double*    v     = (double*)&mask4;

      // printf("%f %f %f %f\n", v[0], v[1], v[2], v[3]);

      // std::cout << "I dont know what is happening. Help me debug this please" << std::endl;

      VectorType mask5 = compare_vector(waveSpeeds0, zeroTol_vec, _CMP_GT_OQ);

      VectorType cmp6  = compare_vector(waveSpeeds0, mul_vector(zeroTol_vec, set_vector(-1.0)), _CMP_GE_OQ);
      VectorType cmp7  = compare_vector(waveSpeeds0, zeroTol_vec, _CMP_LE_OQ);
      VectorType mask6 = bitwise_and(cmp6, cmp7);

      VectorType temp1 = add_vector(o_hUpdateLeft_vec, fWaves0);

      // std::cout << "Mom come pick me up. I am scared. I am at mask4, line 299" << std::endl;

      // v = (double*)&mask4;
      // printf("%f %f %f %f\n", v[0], v[1], v[2], v[3]);

      o_hUpdateLeft_vec = blend_vector(o_hUpdateLeft_vec, temp1, mask4);

      o_huUpdateLeft_vec = blend_vector(
        o_huUpdateLeft_vec, add_vector(o_huUpdateLeft_vec, mul_vector(fWaves0, waveSpeeds0)), mask4
      );

      o_hUpdateRight_vec  = blend_vector(o_hUpdateRight_vec, add_vector(o_hUpdateRight_vec, fWaves0), mask5);
      o_huUpdateRight_vec = blend_vector(
        o_huUpdateRight_vec, add_vector(o_huUpdateRight_vec, mul_vector(fWaves0, waveSpeeds0)), mask5
      );

      o_hUpdateLeft_vec = blend_vector(
        o_hUpdateLeft_vec, add_vector(o_hUpdateLeft_vec, mul_vector(set_vector(0.5), fWaves0)), mask6
      );
      // print_mm256d(o_hUpdateLeft);
      o_huUpdateLeft_vec = blend_vector(
        o_huUpdateLeft_vec,
        add_vector(o_huUpdateLeft_vec, mul_vector(set_vector(0.5), mul_vector(fWaves0, waveSpeeds0))),
        mask6
      );
      o_hUpdateRight_vec = blend_vector(
        o_hUpdateRight_vec, add_vector(o_hUpdateRight_vec, mul_vector(set_vector(0.5), fWaves0)), mask6
      );
      o_huUpdateRight_vec = blend_vector(
        o_huUpdateRight_vec,
        add_vector(o_huUpdateRight_vec, mul_vector(set_vector(0.5), mul_vector(fWaves0, waveSpeeds0))),
        mask6
      );

      VectorType mask7 = compare_vector(waveSpeeds1, mul_vector(zeroTol_vec, set_vector(-1.0)), _CMP_LT_OQ);
      VectorType mask8 = compare_vector(waveSpeeds1, zeroTol_vec, _CMP_GT_OQ);

      VectorType cmp8  = compare_vector(waveSpeeds1, mul_vector(zeroTol_vec, set_vector(-1.0)), _CMP_GE_OQ);
      VectorType cmp9  = compare_vector(waveSpeeds1, zeroTol_vec, _CMP_LE_OQ);
      VectorType mask9 = bitwise_and(cmp8, cmp9);

      o_hUpdateLeft_vec  = blend_vector(o_hUpdateLeft_vec, add_vector(o_hUpdateLeft_vec, fWaves1), mask7);
      o_huUpdateLeft_vec = blend_vector(
        o_huUpdateLeft_vec, add_vector(o_huUpdateLeft_vec, mul_vector(fWaves1, waveSpeeds1)), mask7
      );

      o_hUpdateRight_vec  = blend_vector(o_hUpdateRight_vec, add_vector(o_hUpdateRight_vec, fWaves1), mask8);
      o_huUpdateRight_vec = blend_vector(
        o_huUpdateRight_vec, add_vector(o_huUpdateRight_vec, mul_vector(fWaves1, waveSpeeds1)), mask8
      );

      o_hUpdateLeft_vec = blend_vector(
        o_hUpdateLeft_vec, add_vector(o_hUpdateLeft_vec, mul_vector(set_vector(0.5), fWaves1)), mask9
      );
      o_huUpdateLeft_vec = blend_vector(
        o_huUpdateLeft_vec,
        add_vector(o_huUpdateLeft_vec, mul_vector(set_vector(0.5), mul_vector(fWaves1, waveSpeeds1))),
        mask9
      );
      o_hUpdateRight_vec = blend_vector(
        o_hUpdateRight_vec, add_vector(o_hUpdateRight_vec, mul_vector(set_vector(0.5), fWaves1)), mask9
      );
      o_huUpdateRight_vec = blend_vector(
        o_huUpdateRight_vec,
        add_vector(o_huUpdateRight_vec, mul_vector(set_vector(0.5), mul_vector(fWaves1, waveSpeeds1))),
        mask9
      );

      VectorType absWaveSpeeds0 = max_vector(waveSpeeds0, mul_vector(waveSpeeds0, set_vector(-1.0)));
      VectorType absWaveSpeeds1 = max_vector(waveSpeeds1, mul_vector(waveSpeeds1, set_vector(-1.0)));

      VectorType o_maxWaveSpeed_vec = max_vector(absWaveSpeeds0, absWaveSpeeds1);

      // for (size_t i = 0; i < VectorLength; i++) {
      // std::cout << o_hUpdateLeft_vec[i] << std::endl;
      // }

      store_vector(o_hUpdateLeft, o_hUpdateLeft_vec);
      store_vector(o_hUpdateRight, o_hUpdateRight_vec);
      store_vector(o_huUpdateLeft, o_huUpdateLeft_vec);
      store_vector(o_huUpdateRight, o_huUpdateRight_vec);

      for (size_t i = 0; i < VectorLength; ++i) {
        o_maxWaveSpeed = std::max(o_maxWaveSpeed, o_maxWaveSpeed_vec[i]);
      }
    }

#ifdef ENABLE_VECTORIZATION
#pragma omp declare simd
#endif
    void fWaveComputeWaveSpeeds(
      const T hLeft,
      const T hRight,
      const T huLeft,
      const T huRight,
      const T uLeft,
      const T uRight,
      const T bLeft,
      const T bRight,
      T&      o_waveSpeed0,
      T&      o_waveSpeed1
    ) const {
      // Helper variables for sqrt of h:
      T sqrtHLeft  = std::sqrt(hLeft);  // 1 FLOP (sqrt)
      T sqrtHRight = std::sqrt(hRight); // 1 FLOP (sqrt)
      // Compute eigenvalues of the jacobian matrices in states Q_{i-1} and Q_{i}
      T characteristicSpeed0 = uLeft - sqrtGravity_ * sqrtHLeft;   // 2 FLOPs
      T characteristicSpeed1 = uRight + sqrtGravity_ * sqrtHRight; // 2 FLOPs

      // Compute "Roe averages"
      T hRoe = T(0.5) * (hRight + hLeft); // 2 FLOPs

      T sqrtHRoe = std::sqrt(hRoe);                        // 1 FLOP (sqrt)
      T uRoe     = uLeft * sqrtHRoe + uRight * sqrtHRight; // 3 FLOPs
      uRoe /= sqrtHLeft + sqrtHRight;                      // 2 FLOPs (1 div)

      // Compute "Roe speeds" from Roe averages
      T roeSpeed0 = uRoe - sqrtGravity_ * sqrtHRoe; // 2 FLOPs
      T roeSpeed1 = uRoe + sqrtGravity_ * sqrtHRoe; // 2 FLOPs

      // Compute Eindfeldt speeds (returned as output parameters)
      o_waveSpeed0 = std::min(characteristicSpeed0, roeSpeed0); // 1 FLOP (min)
      o_waveSpeed1 = std::max(characteristicSpeed1, roeSpeed1); // 1 FLOP (max)

      // ==============
      // 20 FLOPs (incl. 3 sqrt, 1 div, 2 min/max)
    }

    void fWaveComputeWaveSpeeds(
      const VectorType hLeft,
      const VectorType hRight,
      const VectorType huLeft,
      const VectorType huRight,
      const VectorType uleft,
      const VectorType uRight,
      const VectorType bLeft,
      const VectorType bRight,
      VectorType&      o_waveSpeed0,
      VectorType&      o_waveSpeed1
    ) const {
      VectorType sqrtHLeft  = square_root_vector(hLeft);
      VectorType sqrtHRight = square_root_vector(hRight);

      VectorType grav                 = set_vector(sqrtGravity_);
      VectorType characteristicSpeed0 = sub_vector(uleft, mul_vector(grav, sqrtHLeft));
      VectorType characteristicSpeed1 = add_vector(uRight, mul_vector(grav, sqrtHRight));

      VectorType hRoe     = mul_vector(set_vector(0.5), add_vector(hRight, hLeft));
      VectorType sqrtHRoe = square_root_vector(hRoe);
      VectorType uRoe     = add_vector(mul_vector(uleft, sqrtHRoe), mul_vector(uRight, sqrtHRight));

      uRoe = div_vector(uRoe, add_vector(sqrtHLeft, sqrtHRight));

      VectorType roeSpeed0 = sub_vector(uRoe, mul_vector(grav, sqrtHRoe));
      VectorType roeSpeed1 = add_vector(uRoe, mul_vector(grav, sqrtHRoe));

      o_waveSpeed0 = min_vector(characteristicSpeed0, roeSpeed0);
      o_waveSpeed1 = max_vector(characteristicSpeed1, roeSpeed1);
    }

#ifdef ENABLE_VECTORIZATION
#pragma omp declare simd
#endif
    void fWaveComputeWaveDecomposition(
      const T hLeft,
      const T hRight,
      const T huLeft,
      const T huRight,
      const T uLeft,
      const T uRight,
      const T bLeft,
      const T bRight,
      const T waveSpeed0,
      const T waveSpeed1,
      T&      o_fWave0,
      T&      o_fWave1
    ) const {
      // Calculate modified (bathymetry) flux difference
      // f(Q_i) - f(Q_{i-1}) -> serve as right hand sides
      T fDif0 = huRight - huLeft; // 1 FLOP
      T fDif1 = huRight * uRight + halfGravity_ * hRight * hRight
                - (huLeft * uLeft + halfGravity_ * hLeft * hLeft); // 9 FLOPs

      // \delta x \Psi[2]
      fDif1 += halfGravity_ * (hRight + hLeft) * (bRight - bLeft); // 5 FLOPs

      // Solve linear system of equations to obtain f-waves:
      // (       1            1      ) ( o_fWave0 ) = ( fDif0 )
      // ( waveSpeed0     waveSpeed1 ) ( o_fWave1 )   ( fDif1 )

      // Compute the inverse of the wave speed difference:
      T inverseSpeedDiff = T(1.0) / (waveSpeed1 - waveSpeed0); // 2 FLOPs (1 div)
      // Compute f-waves:
      o_fWave0 = (waveSpeed1 * fDif0 - fDif1) * inverseSpeedDiff;  // 3 FLOPs
      o_fWave1 = (-waveSpeed0 * fDif0 + fDif1) * inverseSpeedDiff; // 3 FLOPs

      // =========
      // 23 FLOPs in total (incl. 1 div)
    }

    void fWaveComputeWaveDecomposition(
      const VectorType hLeft,
      const VectorType hRight,
      const VectorType huLeft,
      const VectorType huRight,
      const VectorType uLeft,
      const VectorType uRight,
      const VectorType bLeft,
      const VectorType bRight,
      const VectorType waveSpeed0,
      const VectorType waveSpeed1,
      VectorType&      o_fWave0,
      VectorType&      o_fWave1
    ) const {
      // Calculate modified (bathymetry) flux difference
      // f(Q_i) - f(Q_{i-1}) -> serve as right hand sides
      // T fDif0 = huRight - huLeft; // 1 FLOP

      VectorType fDif0 = sub_vector(huRight, huLeft);

      // T fDif1 = huRight * uRight + halfGravity_ * hRight * hRight
      //           - (huLeft * uLeft + halfGravity_ * hLeft * hLeft); // 9 FLOPs

      VectorType grav = set_vector(halfGravity_);

      VectorType fDif1 = sub_vector(
        add_vector(mul_vector(huRight, uRight), mul_vector(grav, mul_vector(hRight, hRight))),
        add_vector(mul_vector(huLeft, uLeft), mul_vector(grav, mul_vector(hLeft, hLeft)))
      );

      // \delta x \Psi[2]
      // fDif1 += halfGravity_ * (hRight + hLeft) * (bRight - bLeft); // 5 FLOPs

      fDif1 = add_vector(fDif1, mul_vector(grav, mul_vector(add_vector(hRight, hLeft), sub_vector(bRight, bLeft))));

      // Solve linear system of equations to obtain f-waves:
      // (       1            1      ) ( o_fWave0 ) = ( fDif0 )
      // ( waveSpeed0     waveSpeed1 ) ( o_fWave1 )   ( fDif1 )

      // Compute the inverse of the wave speed difference:

      VectorType inverseSpeedDiff = div_vector(set_vector(1.0), sub_vector(waveSpeed1, waveSpeed0));

      // T inverseSpeedDiff = T(1.0) / (waveSpeed1 - waveSpeed0); // 2 FLOPs (1 div)
      // Compute f-waves:
      // o_fWave0 = (waveSpeed1 * fDif0 - fDif1) * inverseSpeedDiff;  // 3 FLOPs
      // o_fWave1 = (-waveSpeed0 * fDif0 + fDif1) * inverseSpeedDiff; // 3 FLOPs

      o_fWave0 = mul_vector(sub_vector(mul_vector(waveSpeed1, fDif0), fDif1), inverseSpeedDiff);
      o_fWave1 = mul_vector(sub_vector(fDif1, mul_vector(waveSpeed0, fDif0)), inverseSpeedDiff);

      // =========
      // 23 FLOPs in total (incl. 1 div)
    }
  };
} // namespace Solvers
