cmake_minimum_required(VERSION 3.16)
project(approx_psi_full_merged CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# volePSI installs its CMake package under either lib/cmake/volePSI (upstream
# spelling) or lib/cmake/volepsi (some packaged builds).  Resolve whichever the
# local install provides instead of hard-coding one spelling.
macro(approx_psi_load_volepsi)
  if(NOT TARGET visa::volePSI)
    set(_volepsi_loaded FALSE)
    foreach(_root IN ITEMS "$ENV{VOLEPSI_ROOT}" "$ENV{HOME}/volepsi/out/install/linux")
      if(NOT _volepsi_loaded AND NOT _root STREQUAL "")
        foreach(_dir IN ITEMS "lib/cmake/volePSI" "lib/cmake/volepsi")
          if(NOT _volepsi_loaded AND EXISTS "${_root}/${_dir}/volePSIConfig.cmake")
            list(APPEND CMAKE_PREFIX_PATH "${_root}")
            include("${_root}/${_dir}/volePSIConfig.cmake")
            set(_volepsi_loaded TRUE)
          endif()
        endforeach()
      endif()
    endforeach()
    if(NOT _volepsi_loaded)
      message(FATAL_ERROR
        "volePSI not found. Set VOLEPSI_ROOT to an install prefix that contains "
        "lib/cmake/volePSI/volePSIConfig.cmake.")
    endif()
  endif()
endmacro()

set(PLAIN_SOURCES
    src/params.cpp
    src/hamming.cpp
    src/projection.cpp
    src/dataset.cpp
    src/approx_psi.cpp
)

add_executable(approx_psi_plain
    src/main.cpp
    ${PLAIN_SOURCES}
)
target_include_directories(approx_psi_plain PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
  target_compile_options(approx_psi_plain PRIVATE /W4)
else()
  target_compile_options(approx_psi_plain PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()

add_executable(approx_psi_tests
    tests/test_main.cpp
    ${PLAIN_SOURCES}
)
target_include_directories(approx_psi_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME approx_psi_tests COMMAND approx_psi_tests)

option(APPROX_PSI_BUILD_SECURE "Build secure-subprotocol layer and fallback tests" ON)
option(APPROX_PSI_HAS_EMP "Compile EMP-backed branches (requires EMP headers/libs configured externally)" OFF)
option(APPROX_PSI_HAS_VOLEPSI "Compile volepsi-backed branches (adapter is still incomplete in this reproduction)" OFF)

if(APPROX_PSI_BUILD_SECURE)
    add_library(approx_psi_secure
        ${PLAIN_SOURCES}
        src/secure/common/share.cpp
        src/secure/emp/emp_context.cpp
        src/secure/ss_hamming.cpp
        src/secure/ss_vmult.cpp
        src/secure/ss_psi.cpp
        src/secure/ss_psi_volepsi.cpp
        src/secure/secure_approx_psi.cpp
        src/secure/volepsi_adapter.cpp
    )
    target_include_directories(approx_psi_secure PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

    if(APPROX_PSI_HAS_EMP)
      find_package(emp-tool REQUIRED)
      find_package(emp-ot REQUIRED)
      find_package(emp-sh2pc REQUIRED)

      target_compile_definitions(approx_psi_secure PUBLIC APPROX_PSI_HAS_EMP=1)
      target_compile_features(approx_psi_secure PUBLIC cxx_std_20)
      if(NOT MSVC)
        target_compile_options(approx_psi_secure PUBLIC -maes -msse4.1)
      endif()
      target_link_libraries(approx_psi_secure PUBLIC ${EMP-SH2PC_LIBRARIES})
    endif()
    if(APPROX_PSI_HAS_VOLEPSI)
      approx_psi_load_volepsi()

      target_compile_definitions(approx_psi_secure PUBLIC APPROX_PSI_HAS_VOLEPSI=1)
      target_link_libraries(approx_psi_secure PUBLIC visa::volePSI)
    endif()

    foreach(test_name IN ITEMS secret_share ss_hamming ss_vmult ss_psi secure_approxpsi)
      add_executable(test_${test_name} tests/test_${test_name}.cpp)
      target_link_libraries(test_${test_name} PRIVATE approx_psi_secure)
      target_include_directories(test_${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
      add_test(NAME secure_${test_name} COMMAND test_${test_name})
    endforeach()
endif()

# -----------------------------------------------------------------------------
# Real two-process EMP SS-Hamming demo (EMP v0.3.x API)
# -----------------------------------------------------------------------------
# This target is deliberately separate from APPROX_PSI_HAS_EMP.  It lets us
# validate a real Alice/Bob garbled-circuit execution before replacing the
# fallback implementation inside SSHamming.
option(APPROX_PSI_BUILD_EMP_HAMMING_DEMO
       "Build real Alice/Bob EMP SS-Hamming demo (requires emp-sh2pc v0.3.x)"
       OFF)

if(APPROX_PSI_BUILD_EMP_HAMMING_DEMO)
    find_package(emp-tool REQUIRED)
    find_package(emp-ot REQUIRED)
    find_package(emp-sh2pc REQUIRED)

    add_executable(ss_hamming_emp
        secure_demo/ss_hamming_emp.cpp
    )

    target_compile_features(ss_hamming_emp PRIVATE cxx_std_20)
    target_link_libraries(ss_hamming_emp PRIVATE
        ${EMP-SH2PC_LIBRARIES}
    )

    if(MSVC)
        target_compile_options(ss_hamming_emp PRIVATE /W4)
    else()
        target_compile_options(ss_hamming_emp PRIVATE -Wall -Wextra -Wpedantic -maes -msse4.1)
    endif()
endif()

# -----------------------------------------------------------------------------
# End-to-end Approx-PSI demo: two processes, real EMP circuit + RsCpsi backend
# -----------------------------------------------------------------------------
option(APPROX_PSI_BUILD_FULL_DEMO
       "Build the end-to-end two-process Approx-PSI demo"
       OFF)

if(APPROX_PSI_BUILD_FULL_DEMO)
    find_package(emp-tool REQUIRED)
    find_package(emp-ot REQUIRED)
    find_package(emp-sh2pc REQUIRED)

    approx_psi_load_volepsi()

    add_executable(approx_psi_full
        secure_demo/approx_psi_full.cpp
        ${PLAIN_SOURCES}
        src/secure/common/share.cpp
        src/secure/emp/emp_context.cpp
        src/secure/ss_hamming.cpp
        src/secure/ss_vmult.cpp
        src/secure/ss_psi.cpp
        src/secure/ss_psi_volepsi.cpp
        src/secure/secure_approx_psi.cpp
    )
    target_include_directories(approx_psi_full PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(approx_psi_full PRIVATE
        APPROX_PSI_HAS_EMP=1
        APPROX_PSI_HAS_VOLEPSI=1
    )
    target_compile_features(approx_psi_full PRIVATE cxx_std_20)
    target_link_libraries(approx_psi_full PRIVATE
        ${EMP-SH2PC_LIBRARIES}
        visa::volePSI
    )

    if(MSVC)
        target_compile_options(approx_psi_full PRIVATE /W4)
    else()
        target_compile_options(approx_psi_full PRIVATE -Wall -Wextra -Wpedantic -maes -msse4.1)
    endif()
endif()

# Secure targets use C++20 and imported EMP targets only when enabled.
if(TARGET approx_psi_secure)
  target_compile_features(approx_psi_secure PUBLIC cxx_std_20)
endif()
if(TARGET approx_psi_full)
  target_compile_features(approx_psi_full PRIVATE cxx_std_20)
endif()

# -----------------------------------------------------------------------------
# Circuit-shape micro-benchmark (AND gates + EMP bytes per candidate)
# -----------------------------------------------------------------------------
option(APPROX_PSI_BUILD_CIRCUIT_BENCH
       "Build the EMP Hamming-circuit micro-benchmark"
       OFF)

if(APPROX_PSI_BUILD_CIRCUIT_BENCH)
  find_package(emp-tool REQUIRED CONFIG)
  find_package(emp-ot REQUIRED CONFIG)
  find_package(emp-sh2pc REQUIRED CONFIG)
  add_executable(bench_circuit tests/bench_circuit.cpp)
  target_include_directories(bench_circuit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_features(bench_circuit PRIVATE cxx_std_20)
  target_link_libraries(bench_circuit PRIVATE
    emp-sh2pc::emp-sh2pc emp-ot::emp-ot emp-tool::emp-tool)
  if(NOT MSVC)
    target_compile_options(bench_circuit PRIVATE -maes -msse4.1)
  endif()
endif()

if(APPROX_PSI_HAS_EMP OR APPROX_PSI_BUILD_FULL_DEMO OR APPROX_PSI_BUILD_EMP_HAMMING_DEMO)
  find_package(emp-tool REQUIRED CONFIG)
  find_package(emp-ot REQUIRED CONFIG)
  find_package(emp-sh2pc REQUIRED CONFIG)
  if(TARGET approx_psi_secure)
    target_link_libraries(approx_psi_secure PUBLIC emp-sh2pc::emp-sh2pc emp-ot::emp-ot emp-tool::emp-tool)
  endif()
  if(TARGET approx_psi_full)
    target_link_libraries(approx_psi_full PRIVATE emp-sh2pc::emp-sh2pc emp-ot::emp-ot emp-tool::emp-tool)
  endif()
endif()
