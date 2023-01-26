include(FetchContent)

FetchContent_Declare(
    SWE-Solvers
    GIT_REPOSITORY git@gitlab.lrz.de:ge26cet/swe-solvers-group2.git
    GIT_TAG        mpirma
)
FetchContent_MakeAvailable(SWE-Solvers)
