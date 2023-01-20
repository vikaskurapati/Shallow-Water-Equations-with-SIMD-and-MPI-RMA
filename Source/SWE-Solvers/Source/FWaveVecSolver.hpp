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

#include <cmath>
#include <iostream>

#include "Tools/RealType.hpp"

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

    void computeNetUpdates(
      VectorType  hLeft,
      VectorType  hRight,
      VectorType  huLeft,
      VectorType  huRight,
      VectorType  bLeft,
      VectorType  bRight,
      VectorType& o_hUpdateLeft,
      VectorType& o_hUpdateRight,
      VectorType& o_huUpdateLeft,
      VectorType& o_huUpdateRight,
      VectorType& o_maxWaveSpeed
    ) const {
      __m256d mask1 = mm256_cmp_pd(hLeft, dryTol_, _CMP_GE_OQ);
      __m256d mask2 = mm256_cmp_pd(hRight, dryTol_, _CMP_LT_OQ);
      __m256d mask3 = mm256_cmp_pd(hRight, dryTol_, _CMP_GE_OQ);

      __m256d mask4 = _mm256_and_pd(mask1, mask2);
      __m256d mask5 = _mm256_andnot_pd(mask1, mask3);
      __m256d mask6 = _mm256_andnot_pd(mask4, mask5);

      // Dry/Wet case
    __m256d dry_wet_hRight = _mm256_and_pd(mask4, hLeft);
    __m256d dry_wet_huRight = _mm256_and_pd(mask4, _mm256_mul_pd(huLeft, _mm256_set1_pd(-1)));
    __m256d dry_wet_bRight = _mm256_and_pd(mask4, bLeft);

    // Wet/Dry case
    __m256d wet_dry_hLeft = _mm256_and_pd(mask5, hRight);
    __m256d wet_dry_huLeft = _mm256_and_pd(mask5, _mm256_mul_pd(huRight, _mm256_set1_pd(-1)));
    __m256d wet_dry_bLeft = _mm256_and_pd(mask5, bRight); 


    // Dry/Dry case
  __m256d dry_dry_hLeft = _mm256_and_pd(mask6, mm256_set1_pd(dryTol_));
  __m256d dry_dry_huLeft = _mm256_and_pd(mask6, _mm256_set1_pd(0.0));
  __m256d dry_dry_bLeft = _mm256_and_pd(mask6, _mm256_set1_pd(0.0));
  __m256d dry_dry_hRight = _mm256_and_pd(mask6, mm256_set1_pd(dryTol_));
  __m256d dry_dry_huRight = _mm256_and_pd(mask6, _mm256_set1_pd(0.0));
  __m256d dry_dry_bRight = _mm256_and_pd(mask6, _mm256_set1_pd(0.0));   

  __m256d uLeft = _mm256_div_pd(huLeft, hLeft);
  __m256d uRight = _mm256_div_pd(huRight, hRight);
  // hRight = _mm256_or_pd(dry_wet_hRight, _mm256_or_pd(wet_dry_hRight, dry_dry_hRight));
  // huRight = _mm256_or_pd(dry_wet_huRight, _mm256_or_pd(wet_dry_huRight, dry_dry_huRight));
  // bRight = _mm256_or_pd(dry_wet_bRight, mm256_or_pd(wet_dry_bRight, dry_dry_bRight));
  __m256d waveSpeeds0 = _mm256_setzero_pd();
  __m256d waveSpeeds1 = _mm256_setzero_pd();
  __m256d fWaves0 = _mm256_setzero_pd();
  __m256d fWaves1 = _mm256_setzero_pd();
    
  fWaveComputeWaveDecomposition(
        hLeft, hRight, huLeft, huRight, uLeft, uRight, bLeft, bRight, waveSpeeds0, waveSpeeds1, fWaves0, fWaves1
      );

// Compute the net-updates
o_hUpdateLeft = _mm256_setzero_pd();
o_hUpdateRight = _mm256_setzero_pd();
o_huUpdateLeft = _mm256_setzero_pd();
o_huUpdateRight = _mm256_setzero_pd();

// 1st wave family
__m256d waveSpeeds0_lt = _mm256_cmp_pd(waveSpeeds0, _mm256_set1_pd(-zeroTol_), _CMP_LT_OQ);
__m256d waveSpeeds0_gt = _mm256_cmp_pd(waveSpeeds0, _mm256_set1_pd(zeroTol_), _CMP_GT_OQ);
__m256d update_left = _mm256_and_pd(fWaves0, waveSpeeds0_lt);
__m256d update_right = _mm256_and_pd(fWaves0, waveSpeeds0_gt);
__m256d update_split = _mm256_andnot_pd(waveSpeeds0_lt, waveSpeeds0_gt);
update_left = _mm256_add_pd(update_left, _mm256_mul_pd(update_split, _mm256_set1_pd(0.5)));
update_right = _mm256_add_pd(update_right, _mm256_mul_pd(update_split, _mm256_set1_pd(0.5)));
o_hUpdateLeft = _mm256_add_pd(o_hUpdateLeft, update_left);
o_huUpdateLeft = _mm256_add_pd(o_huUpdateLeft, _mm256_mul_pd(update_left, waveSpeeds0));
o_hUpdateRight = _mm256_add_pd(o_hUpdateRight, update_right);
o_huUpdateRight = _mm256_add_pd(o_huUpdateRight, _mm256_mul_pd(update_right, waveSpeeds0));


// 2nd Wave family
__m256d zeroTol_v = mm256_set1_pd(zeroTol_);
__m256d half_v = _mm256_set1_pd(0.5);

__m256d cmpRight_v = _mm256_cmp_pd(waveSpeeds1, zeroTol_v, _CMP_GT_OQ);
__m256d cmpLeft_v = _mm256_cmp_pd(waveSpeeds1, zeroTol_v, _CMP_LT_OQ);


o_hUpdateRight = _mm256_add_pd(o_hUpdateRight, _mm256_and_pd(cmpRight_v, fWaves1));
o_huUpdateRight = _mm256_add_pd(o_huUpdateRight, _mm256_mul_pd(_mm256_and_pd(cmpRight_v, fWaves1), waveSpeeds1));
o_hUpdateLeft = _mm256_add_pd(o_hUpdateLeft, _mm256_and_pd(cmpLeft_v, fWaves1));
o_huUpdateLeft = _mm256_add_pd(o_huUpdateLeft, _mm256_mul_pd(_mm256_and_pd(cmpLeft_v, fWaves1), waveSpeeds1));

__m256d o_hUpdateSplit_v = _mm256_mul_pd(half_v, fWaves1);
__m256d o_huUpdateSplit_v = _mm256_mul_pd(_mm256_mul_pd(half_v, fWaves1), waveSpeeds1);

o_hUpdateRight = _mm256_add_pd(o_hUpdateRight, _mm256_andnot_pd(cmpRight_v, o_hUpdateSplit_v));
o_huUpdateRight = _mm256_add_pd(o_huUpdateRight, _mm256_andnot_pd(cmpRight_v, o_huUpdateSplit_v));
o_hUpdateLeft = _mm256_add_pd(o_hUpdateLeft, _mm256_andnot_pd(cmpLeft_v, o_hUpdateSplit_v));
o_huUpdateLeft = _mm256_add_pd(o_huUpdateLeft, _mm256_andnot_pd(cmpLeft_v, o_huUpdateSplit_v));


__m256d absWaveSpeeds0_v = _mm256_and_pd(waveSpeeds0, _mm256_set1_pd(-0.0));
__m256d absWaveSpeeds1_v = _mm256_and_pd(waveSpeeds1, _mm256_set1_pd(-0.0));

__m256d maxWaveSpeeds_v = _mm256_max_pd(absWaveSpeeds0_v, absWaveSpeeds1_v);
o_maxWaveSpeed = maxWaveSpeeds_v;

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
