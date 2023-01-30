/**
 * @file
 * This file is part of SWE.
 *
 * @author Michael Bader (bader AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Univ.-Prof._Dr._Michael_Bader)
 * @author Alexander Breuer (breuera AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Dipl.-Math._Alexander_Breuer)
 * @author Sebastian Rettenberger (rettenbs AT in.tum.de,
 * http://www5.in.tum.de/wiki/index.php/Sebastian_Rettenberger,_M.Sc.)
 *
 * @section LICENSE
 *
 * SWE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SWE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SWE.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * @section DESCRIPTION
 *
 * Setting of SWE, which uses a wave propagation solver and an artificial or ASAGI scenario on multiple blocks.
 */

#include <cmath>
#include <csignal>
#include <fenv.h>
#include <mpi.h>

#include "Blocks/Block.hpp"
#include "Blocks/WavePropagationBlock.hpp"
#include "Scenarios/BathymetryDamBreakScenario.hpp"
#include "Scenarios/RadialDamBreakScenario.hpp"
#include "Scenarios/SeaAtRestScenario.hpp"
#include "Scenarios/SplashingConeScenario.hpp"
#include "Scenarios/SplashingPoolScenario.hpp"
#include "Tools/Args.hpp"
#include "Tools/Logger.hpp"
#include "Tools/ProgressBar.hpp"
#include "Writers/Writer.hpp"

#ifndef _MSC_VER
#pragma float_control(precise, on)
#pragma STDC FENV_ACCESS ON
#endif

#ifdef _MSC_VER
void fpExceptionHandler(const int signal, const int nSubCode);
#endif

/**
 * Computes the number of block rows from the total number of processes.
 *
 * The number of rows is determined as the square root of the
 * number of processes, if this is a square number;
 * otherwise, we use the largest number that is smaller than the square
 * root and still a divisor of the number of processes.
 *
 * @param numberOfProcesses number of processes
 * @return number of block rows
 */
int computeNumberOfBlockRows(int numberOfProcesses);

void exchangeNeighborInfo(int* currentinfo, int* neighborInfo, int mpiRank, int topNeighborRank, int bottomNeighborRank);

void exchangeLayers(
  const int    leftNeighborRank,
  double*      o_leftInflow,
  double*      leftOutflow,
  const int    rightNeighborRank,
  double*      o_rightInflow,
  double*      rightOutflow,
  MPI_Datatype mpiCol,
  int          len,
  MPI_Datatype mpiCol2
);

int main(int argc, char** argv) {
  //! MPI Rank of a process.
  int mpiRank = -1;
  //! Number of MPI processes.
  int numberOfProcesses = -1;
  // Initialize MPI
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cerr << "MPI_Init failed." << std::endl;
    return EXIT_FAILURE;
  }

  // Determine local MPI rank
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  // Determine total number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);

  Tools::Logger::logger.setProcessRank(mpiRank);
  Tools::Logger::logger.printWelcomeMessage();
  Tools::Logger::logger.printNumberOfProcesses(numberOfProcesses);

#ifndef _MSC_VER
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#else
  unsigned int oldState = 0;
  _controlfp_s(&oldState, _MCW_EM, _MCW_EM);
  const unsigned int flags      = _EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW;
  const unsigned int enableBits = flags & _MCW_EM;
  _clearfp();
  _controlfp_s(0, ~enableBits, enableBits);
  // std::signal(SIGFPE, (void(__cdecl*)(int)) fpExceptionHandler); // Gives better printing, but hides the actual
  // location of the floating point error.
#endif

  Tools::Args args;
  args.addOption("grid-size-x", 'x', "Number of cells in x direction");
  args.addOption("grid-size-y", 'y', "Number of cells in y direction");
  args.addOption("output-basepath", 'o', "Output base file name");
  args.addOption("number-of-checkpoints", 'n', "Number of checkpoints to write output files");

  Tools::Args::Result ret = args.parse(argc, argv, mpiRank == 0);

  switch (ret) {
  case Tools::Args::Result::Error:
    MPI_Abort(MPI_COMM_WORLD, -1);
    return EXIT_FAILURE;
  case Tools::Args::Result::Help:
    MPI_Finalize();
    return EXIT_SUCCESS;
  default:
    break;
  }
  int mpirow_len, mpicol_len, col_disp, row_disp;

  int         numberOfGridCellsX  = args.getArgument<int>("grid-size-x", 16);
  int         numberOfGridCellsY  = args.getArgument<int>("grid-size-y", 16);
  std::string baseName            = args.getArgument<std::string>("output-basepath", "SWE");
  int         numberOfCheckPoints = args.getArgument<int>(
    "number-of-checkpoints", 20
  ); //! Number of checkpoints for visualization (at each checkpoint in time, an output file is written).

  // Print information about the grid
  Tools::Logger::logger.printNumberOfCells(numberOfGridCellsX, numberOfGridCellsY);

  // Determine the layout of MPI-ranks: use numberOfBlocksY*numberOfBlocksX grid blocks

  int numberOfBlocksY = computeNumberOfBlockRows(numberOfProcesses);

  int numberOfBlocksX = numberOfProcesses / numberOfBlocksY;
  Tools::Logger::logger.printNumberOfBlocks(numberOfBlocksX, numberOfBlocksY);

  // Determine local block coordinates of each block

  int blockPositionX = mpiRank / numberOfBlocksY;
  int blockPositionY = mpiRank % numberOfBlocksY;

  // Number of grid cells in x- and y-direction per process
  // Compute local number of cells for each block
  int nXLocal  = (blockPositionX < numberOfBlocksX - 1)
                   ? numberOfGridCellsX / numberOfBlocksX
                   : numberOfGridCellsX - (numberOfBlocksX - 1) * (numberOfGridCellsX / numberOfBlocksX);
  int nYLocal  = (blockPositionY < numberOfBlocksY - 1)
                   ? numberOfGridCellsY / numberOfBlocksY
                   : numberOfGridCellsY - (numberOfBlocksY - 1) * (numberOfGridCellsY / numberOfBlocksY);
  int nXNormal = numberOfGridCellsX / numberOfBlocksX;
  int nYNormal = numberOfGridCellsY / numberOfBlocksY;
  Tools::Logger::logger.printNumberOfCellsPerProcess(nXLocal, nYLocal);

  // Create a simple artificial scenario
  Scenarios::RadialDamBreakScenario scenario;

  // Compute the size of a single cell
  RealType cellSizeX = (scenario.getBoundaryPos(BoundaryEdge::Right) - scenario.getBoundaryPos(BoundaryEdge::Left))
                       / numberOfGridCellsX;
  RealType cellSizeY = (scenario.getBoundaryPos(BoundaryEdge::Top) - scenario.getBoundaryPos(BoundaryEdge::Bottom))
                       / numberOfGridCellsY;
  Tools::Logger::logger.printCellSize(cellSizeX, cellSizeY);

  auto waveBlock = Blocks::Block::getBlockInstance(nXLocal, nYLocal, cellSizeX, cellSizeY);

  // Get the origin from the scenario
  RealType originX = scenario.getBoundaryPos(BoundaryEdge::Left) + blockPositionX * nXNormal * cellSizeX;
  RealType originY = scenario.getBoundaryPos(BoundaryEdge::Bottom) + blockPositionY * nYNormal * cellSizeY;

  // Initialise the wave propagation block
  waveBlock->initialiseScenario(originX, originY, scenario, true);

  // Get the final simulation time from the scenario
  double endSimulationTime = scenario.getEndSimulationTime();

  // Checkpoints when output files are written
  double* checkPoints = new double[numberOfCheckPoints + 1];

  // Compute the checkpoints in time
  for (int cp = 0; cp <= numberOfCheckPoints; cp++) {
    checkPoints[cp] = cp * (endSimulationTime / numberOfCheckPoints);
  }

  /*
   * Connect blocks at boundaries
   */
  // Left and right boundaries
  Tools::Logger::logger.printString("Connecting SWE blocks at left boundaries.");
  Blocks::Block1D* leftInflow  = waveBlock->grabGhostLayer(BoundaryEdge::Left);
  Blocks::Block1D* leftOutflow = waveBlock->registerCopyLayer(BoundaryEdge::Left);
  if (blockPositionX == 0) {
    waveBlock->setBoundaryType(BoundaryEdge::Left, BoundaryType::Outflow);
  }

  Tools::Logger::logger.printString("Connecting SWE blocks at right boundaries.");
  Blocks::Block1D* rightInflow  = waveBlock->grabGhostLayer(BoundaryEdge::Right);
  Blocks::Block1D* rightOutflow = waveBlock->registerCopyLayer(BoundaryEdge::Right);
  if (blockPositionX == numberOfBlocksX - 1) {
    waveBlock->setBoundaryType(BoundaryEdge::Right, BoundaryType::Outflow);
  }

  // Bottom and top boundaries
  Tools::Logger::logger.printString("Connecting SWE blocks at bottom boundaries.");
  Blocks::Block1D* bottomInflow  = waveBlock->grabGhostLayer(BoundaryEdge::Bottom);
  Blocks::Block1D* bottomOutflow = waveBlock->registerCopyLayer(BoundaryEdge::Bottom);
  if (blockPositionY == 0) {
    waveBlock->setBoundaryType(BoundaryEdge::Bottom, BoundaryType::Outflow);
  }

  Tools::Logger::logger.printString("Connecting SWE blocks at top boundaries.");
  Blocks::Block1D* topInflow  = waveBlock->grabGhostLayer(BoundaryEdge::Top);
  Blocks::Block1D* topOutflow = waveBlock->registerCopyLayer(BoundaryEdge::Top);
  if (blockPositionY == numberOfBlocksY - 1) {
    waveBlock->setBoundaryType(BoundaryEdge::Top, BoundaryType::Outflow);
  }

  /*
   * The grid is stored column wise in memory:
   *
   *        ************************** . . . **********
   *        *       *  ny+2 *2(ny+2)*         * (ny+1)*
   *        *  ny+1 * +ny+1 * +ny+1 *         * (ny+2)*
   *        *       *       *       *         * +ny+1 *
   *        ************************** . . . **********
   *        *       *       *       *         *       *
   *        .       .       .       .         .       .
   *        .       .       .       .         .       .
   *        .       .       .       .         .       .
   *        *       *       *       *         *       *
   *        ************************** . . . **********
   *        *       *  ny+2 *2(ny+2)*         * (ny+1)*
   *        *   1   *   +1  *   +1  *         * (ny+2)*
   *        *       *       *       *         *   +1  *
   *        ************************** . . . **********
   *        *       *  ny+2 *2(ny+2)*         * (ny+1)*
   *        *   0   *   +0  *   +0  *         * (ny+2)*
   *        *       *       *       *         *   +0  *
   *        ************************** . . . ***********
   *
   *  -> The stride for a row is ny+2, because we have to jump over a whole column
   *     for every row-element. This holds only in the CPU-version, in CUDA a buffer is implemented.
   *     See Blocks/CUDA/CUDABlock.hpp/.cu for details.
   *  -> The stride for a column is 1, because we can access the elements linear in memory.
   */

  //! MPI row-vector: nXLocal+2 blocks, 1 element per block, stride of nYLocal+2
  MPI_Datatype mpiRow;
#ifndef ENABLE_CUDA
  MPI_Type_vector(nXLocal + 2, 1, nYLocal + 2, MY_MPI_FLOAT, &mpiRow);
#else
  MPI_Type_vector(nXLocal + 2, 1, nYLocal + 2, MY_MPI_FLOAT, &mpiRow);
#endif
  MPI_Type_commit(&mpiRow);
  mpirow_len = (nXLocal + 2) * 1 + (nYLocal + 2) * (nXLocal + 2 - 1);
  row_disp   = nYLocal + 2;
  //! MPI column-vector: 1 block, nYLocal+2 elements per block, stride of 1
  // std::cout<<"\nVector"<<nYLocal+2<<" " <<std::endl;
  // std::cout<<"\nVector"<<nYLocal+2<<" " <<std::endl;
  MPI_Datatype mpiCol;
  MPI_Type_vector(1, nYLocal + 2, 1, MY_MPI_FLOAT, &mpiCol);
  MPI_Type_commit(&mpiCol);
  mpicol_len = 1 * (nYLocal + 2);
  col_disp   = 1;

  
  
  // Compute MPI ranks of the neighbour processes
  int leftNeighborRank   = (blockPositionX > 0) ? mpiRank - numberOfBlocksY : MPI_PROC_NULL;
  int rightNeighborRank  = (blockPositionX < numberOfBlocksX - 1) ? mpiRank + numberOfBlocksY : MPI_PROC_NULL;
  int bottomNeighborRank = (blockPositionY > 0) ? mpiRank - 1 : MPI_PROC_NULL;
  int topNeighborRank    = (blockPositionY < numberOfBlocksY - 1) ? mpiRank + 1 : MPI_PROC_NULL;
  int  localvariables[2] = {nXLocal, nYLocal};
  int* currentinfo             = &localvariables[0];
  int* neighborinfo    = (int*)malloc(4 * sizeof(int));
  exchangeNeighborInfo(currentinfo,neighborinfo,mpiRank,topNeighborRank,bottomNeighborRank);
  MPI_Datatype mpiCol2;

  if (topNeighborRank>=0) {
    MPI_Type_vector(*(neighborinfo+0) + 2, 1, *(neighborinfo+1) + 2, MY_MPI_FLOAT, &mpiCol2);
    MPI_Type_commit(&mpiCol2);
  }
  else if (bottomNeighborRank>=0) {
    MPI_Type_vector(*(neighborinfo+2) + 2, 1, *(neighborinfo+3) + 2, MY_MPI_FLOAT, &mpiCol2);
    MPI_Type_commit(&mpiCol2);
  }
  else{
    mpiCol2 = mpiCol;
  }

  // Print the MPI grid
  Tools::Logger::logger.getDefaultOutputStream()
    << "Neighbors: " << leftNeighborRank << " (left), " << rightNeighborRank << " (right), " << bottomNeighborRank
    << " (bottom), " << topNeighborRank << " (top)" << std::endl;

  exchangeLayers(
    leftNeighborRank,
    leftInflow->h.getData(),
    leftOutflow->h.getData(),
    rightNeighborRank,
    rightInflow->h.getData(),
    rightOutflow->h.getData(),
    mpiCol,
    mpicol_len,
    mpiCol
  );

  exchangeLayers(
    topNeighborRank,
    topInflow->h.getData(),
    topOutflow->h.getData(),
    bottomNeighborRank,
    bottomInflow->h.getData(),
    bottomOutflow->h.getData(),
    mpiRow,
    mpirow_len,
    mpiCol2
  );

  if (mpiRank == 0) {
    for (int i = 0; i < nYLocal + 2; i++) {
      // std::cout << "\nAFTER b[" << i << "] = " << *(rightInflow->h.getData() + i) << std::endl;
    }
  }
  exchangeLayers(
    leftNeighborRank,
    leftInflow->hu.getData(),
    leftOutflow->hu.getData(),
    rightNeighborRank,
    rightInflow->hu.getData(),
    rightOutflow->hu.getData(),
    mpiCol,
    mpicol_len,
    mpiCol
  );
  exchangeLayers(
    topNeighborRank,
    topInflow->hu.getData(),
    topOutflow->hu.getData(),
    bottomNeighborRank,
    bottomInflow->hu.getData(),
    bottomOutflow->hu.getData(),
    mpiRow,
    mpirow_len,
    mpiCol2
  );

  exchangeLayers(
    leftNeighborRank,
    leftInflow->hv.getData(),
    leftOutflow->hv.getData(),
    rightNeighborRank,
    rightInflow->hv.getData(),
    rightOutflow->hv.getData(),
    mpiCol,
    mpicol_len,
    mpiCol
  );
  exchangeLayers(
    topNeighborRank,
    topInflow->hv.getData(),
    topOutflow->hv.getData(),
    bottomNeighborRank,
    bottomInflow->hv.getData(),
    bottomOutflow->hv.getData(),
    mpiRow,
    mpirow_len,
    mpiCol2
  );

  Tools::ProgressBar progressBar(endSimulationTime, mpiRank);

  Tools::Logger::logger.printOutputTime(0.0);
  progressBar.update(0.0);

  // Boundary size of the ghost layers
  //Writers::BoundarySize boundarySize = {{1, 1, 1, 1}};

  // std::string fileName = Writers::generateBaseFileName(baseName, blockPositionX, blockPositionY);
  // auto        writer   = Writers::Writer::createWriterInstance(
  //   fileName,
  //   waveBlock->getBathymetry(),
  //   boundarySize,
  //   nXLocal,
  //   nYLocal,
  //   cellSizeX,
  //   cellSizeY,
  //   blockPositionX * nXLocal,
  //   blockPositionY * nYLocal,
  //   originX,
  //   originY,
  //   0
  // );

  // Write zero time step
  // writer->writeTimeStep(waveBlock->getWaterHeight(), waveBlock->getDischargeHu(), waveBlock->getDischargeHv(), 0.0);

  // Print the start message and reset the wall clock time
  progressBar.clear();
  Tools::Logger::logger.printStartMessage();
  Tools::Logger::logger.initWallClockTime(time(NULL)); // MPI_Wtime()

  double simulationTime = 0.0;
  progressBar.update(simulationTime);

  unsigned int iterations = 0;

  // Loop over checkpoints
  for (int cp = 1; cp <= numberOfCheckPoints; cp++) {
    // Do time steps until next checkpoint is reached
    while (simulationTime < checkPoints[cp]) {
      // Reset CPU-Communication clock
      Tools::Logger::logger.resetClockToCurrentTime("CPU-Communication");

      exchangeLayers(
        leftNeighborRank,
        leftInflow->h.getData(),
        leftOutflow->h.getData(),
        rightNeighborRank,
        rightInflow->h.getData(),
        rightOutflow->h.getData(),
        mpiCol,
        mpicol_len,
        mpiCol
      );

      exchangeLayers(
        topNeighborRank,
        topInflow->h.getData(),
        topOutflow->h.getData(),
        bottomNeighborRank,
        bottomInflow->h.getData(),
        bottomOutflow->h.getData(),
        mpiRow,
        mpirow_len,
        mpiCol2
      );

      exchangeLayers(
        leftNeighborRank,
        leftInflow->hu.getData(),
        leftOutflow->hu.getData(),
        rightNeighborRank,
        rightInflow->hu.getData(),
        rightOutflow->hu.getData(),
        mpiCol,
        mpicol_len,
        mpiCol
      );
      exchangeLayers(
        topNeighborRank,
        topInflow->hu.getData(),
        topOutflow->hu.getData(),
        bottomNeighborRank,
        bottomInflow->hu.getData(),
        bottomOutflow->hu.getData(),
        mpiRow,
        mpirow_len,
        mpiCol2
      );
      exchangeLayers(
        leftNeighborRank,
        leftInflow->hv.getData(),
        leftOutflow->hv.getData(),
        rightNeighborRank,
        rightInflow->hv.getData(),
        rightOutflow->hv.getData(),
        mpiCol,
        mpicol_len,
        mpiCol
      );
      exchangeLayers(
        topNeighborRank,
        topInflow->hv.getData(),
        topOutflow->hv.getData(),
        bottomNeighborRank,
        bottomInflow->hv.getData(),
        bottomOutflow->hv.getData(),
        mpiRow,
        mpirow_len,
        mpiCol2
      );
      // Reset the cpu clock
      Tools::Logger::logger.resetClockToCurrentTime("CPU");

      // Set values in ghost cells
      waveBlock->setGhostLayer();

      //  Compute numerical flux on each edge

      waveBlock->computeNumericalFluxes();

      //  Approximate the maximum time step
      //  waveBlock->computeMaxTimeStep();

      RealType maxTimeStepWidth = waveBlock->getMaxTimeStep();

      //! Maximum allowed time steps of all blocks
      RealType maxTimeStepWidthGlobal = RealType(0.0);

      // Determine smallest time step of all blocks
      MPI_Allreduce(&maxTimeStepWidth, &maxTimeStepWidthGlobal, 1, MY_MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);

      // Update the cell values
      waveBlock->updateUnknowns(maxTimeStepWidthGlobal);

      // Update the cpu time in the logger
      Tools::Logger::logger.updateTime("CPU");
      Tools::Logger::logger.updateTime("CPU-Communication");

      // Print the current simulation time
      progressBar.clear();
      Tools::Logger::logger.printSimulationTime(
        simulationTime,
        "[" + std::to_string(iterations) + "]: Simulation with max. global dt " + std::to_string(maxTimeStepWidthGlobal)
          + " at time"
      );

      // Update simulation time with time step width
      simulationTime += maxTimeStepWidthGlobal;
      iterations++;
      progressBar.update(simulationTime);
    }

    // Print current simulation time of the output
    progressBar.clear();
    Tools::Logger::logger.printOutputTime(simulationTime);
    progressBar.update(simulationTime);

    // Write output
    // writer->writeTimeStep(
    //   waveBlock->getWaterHeight(), waveBlock->getDischargeHu(), waveBlock->getDischargeHv(), simulationTime
    // );
  }

  progressBar.clear();
  Tools::Logger::logger.printStatisticsMessage();
  Tools::Logger::logger.printTime("CPU", "CPU Time");
  Tools::Logger::logger.printTime("CPU-Communication", "CPU + Communication Time");
  Tools::Logger::logger.printWallClockTime(time(NULL));
  Tools::Logger::logger.printIterationsDone(iterations);

  Tools::Logger::logger.printFinishMessage();

  delete waveBlock;
  delete[] checkPoints;

  MPI_Finalize();

  return EXIT_SUCCESS;
}

#ifdef _MSC_VER
void fpExceptionHandler(const int signal, const int nSubCode) {
  (void)signal;
  //_fpreset();
  _clearfp();
  switch (nSubCode) {
  case _FPE_INVALID:
    throw std::logic_error("Invalid number (NaN) encountered");
    break;
  case _FPE_DENORMAL:
    throw std::logic_error("Denormal");
    break;
  case _FPE_ZERODIVIDE:
    throw std::logic_error("Division by zero");
    break;
  case _FPE_OVERFLOW:
    throw std::logic_error("Overflow error encountered");
    break;
  case _FPE_UNDERFLOW:
    throw std::logic_error("Underflow error encountered");
    break;
  case _FPE_INEXACT:
    throw std::logic_error("Inexact floating point operation encountered");
    break;
  default:
    std::stringstream ss;
    ss << "Floating point error with error code " << nSubCode;
    throw std::logic_error(ss.str());
    break;
  }
}
#endif

int computeNumberOfBlockRows(int numberOfProcesses) {
  int numberOfRows = static_cast<int>(std::sqrt(numberOfProcesses));
  while (numberOfProcesses % numberOfRows != 0)
    numberOfRows--;
  return numberOfRows;
}

void exchangeLayers(
  const int    leftNeighborRank,  // Rank of the left neighbor process
  double*      o_leftInflow,      // Inflow buffer for the left neighbor
  double*      leftOutflow,       // Outflow buffer for the left neighbor
  const int    rightNeighborRank, // Rank of the right neighbor process
  double*      o_rightInflow,     // Inflow buffer for the right neighbor
  double*      rightOutflow,      // Outflow buffer for the right neighbor
  MPI_Datatype mpiCol,            // MPI data type for the ghost layer column
  int          mpicol_len,        // Length of the ghost layer column
  MPI_Datatype mpiCol2            // MPI data type for the ghost layer column
) {
  MPI_Win leftWin, rightWin; // Window objects for the left and right ghost layers

  //  create the window for the right ghost layer

  MPI_Win_create(
    o_leftInflow, (mpicol_len) * sizeof(MY_MPI_FLOAT), sizeof(MY_MPI_FLOAT), MPI_INFO_NULL, MPI_COMM_WORLD, &leftWin
  );

  // Synchronize the window before accessing the data
  MPI_Win_fence(0, leftWin);
  MPI_Put(rightOutflow, 1, mpiCol, rightNeighborRank, 0, 1, mpiCol2, leftWin);
  MPI_Win_fence(0, leftWin);

  // Free the window after use
  MPI_Win_free(&leftWin);

  //  create the window for the left ghost layer

  MPI_Win_create(
    o_rightInflow, (mpicol_len) * sizeof(MY_MPI_FLOAT), sizeof(MY_MPI_FLOAT), MPI_INFO_NULL, MPI_COMM_WORLD, &rightWin
  );
  // Synchronize the window before accessing the data
  MPI_Win_fence(0, rightWin);
  MPI_Put(leftOutflow, 1, mpiCol, leftNeighborRank, 0, 1, mpiCol2, rightWin);
  MPI_Win_fence(0, rightWin);

  // Free the window after use
  MPI_Win_free(&rightWin);
}


/*
Exchange neighbor info for non contiguous communication of information
*/
void exchangeNeighborInfo(int* currentinfo, int* neighborInfo, int mpiRank, int topNeighborRank,int bottomNeighborRank) {
  MPI_Win win;
  
  MPI_Win_create(currentinfo, 2 * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
  
    // Synchronize the window before accessing the data
  MPI_Win_fence(0, win);
  MPI_Get(neighborInfo, 2, MPI_INT, topNeighborRank, 0, 2, MPI_INT, win);
  MPI_Get(neighborInfo+2, 2, MPI_INT, bottomNeighborRank, 0, 2, MPI_INT, win);
  MPI_Win_fence(0, win);
  // Free the window after use
  MPI_Win_free(&win);

  // Synchronize the window before accessing the data
  
  // Free the window after use
}
