cmake_minimum_required(VERSION 3.24)
project(chronotope LANGUAGES CXX CUDA)
# -march=native for every host-compiled object (user rule 2026-08-27); nvcc host passes get it via -Xcompiler
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 121)   # GB10; override with -DCMAKE_CUDA_ARCHITECTURES=<sm>
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# host float arithmetic must not be contracted into FMA: the CPU references in the tests mirror numpy's float32 ops
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)

# ---- the core: one set of objects, a STATIC archive (the default link for the binary and the tests: no PLT/GOT
# indirection, whole-archive inlining across the API boundary) and a SHARED library (what the Python referee loads
# through ctypes). -DCHRONOTOPE_BUILD_SHARED=OFF drops the .so; -DCHRONOTOPE_LINK_STATIC=OFF links the binary against
# the .so instead (user rule 2026-08-27: both available, static by default).
option(CHRONOTOPE_BUILD_SHARED "Also build libchronotope.so (needed by the ctypes referee path)" ON)
option(CHRONOTOPE_LINK_STATIC "Link the chronotope binary and tests against libchronotope.a (default) instead of the .so" ON)
option(CHRONOTOPE_HOSTED_API "Build opt-in C++ HTTPS hosted planner (no TensorRT required)" OFF)
set(CHRONOTOPE_EDGELLM_BUILD "" CACHE PATH "TensorRT-Edge-LLM build directory (enables --planner trt and the in-library planner)")
set(CHRONOTOPE_TRT_PREFIX "" CACHE PATH "TensorRT prefix (the extracted debs: <prefix>/usr/...)")
set(CHRONOTOPE_CUDA_ARCH_DIR /usr/local/cuda/targets/sbsa-linux CACHE PATH "CUDA target dir")
set(CHRONOTOPE_CORE_SOURCES src/vision.cu src/control.cu src/core.cu src/chronotope_api.cu)
if(CHRONOTOPE_EDGELLM_BUILD)
  # K4d: the in-process planner is part of the LIBRARY, so the ctypes referee path and the binary make the same call
  list(APPEND CHRONOTOPE_CORE_SOURCES src/planner_trt.cpp src/planner_vlm.cpp)
endif()
add_library(chronotope_objs OBJECT ${CHRONOTOPE_CORE_SOURCES})
target_include_directories(chronotope_objs PUBLIC src)
target_compile_options(chronotope_objs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
set_target_properties(chronotope_objs PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
add_library(chronotope_static STATIC $<TARGET_OBJECTS:chronotope_objs>)
target_include_directories(chronotope_static PUBLIC src)
set_target_properties(chronotope_static PROPERTIES OUTPUT_NAME chronotope)     # libchronotope.a
if(CHRONOTOPE_BUILD_SHARED)
  if(CHRONOTOPE_EDGELLM_BUILD)   # a CUDA source compiled with separable compilation makes CMake device-link the Edge-LLM objects into the .so
    add_library(chronotope SHARED $<TARGET_OBJECTS:chronotope_objs> src/chronotope_devlink_stub.cu)
  else()
    add_library(chronotope SHARED $<TARGET_OBJECTS:chronotope_objs>)           # libchronotope.so
  endif()
  target_include_directories(chronotope PUBLIC src)
endif()
find_package(ZLIB REQUIRED)   # planner_vlm.cpp's PNG encoder (the HTTP planner) lives in the core
target_link_libraries(chronotope_static INTERFACE ZLIB::ZLIB)
if(TARGET chronotope)
  target_link_libraries(chronotope PRIVATE ZLIB::ZLIB)
endif()
if(CHRONOTOPE_LINK_STATIC)
  set(CHRONOTOPE_CORE_LIB chronotope_static)
else()
  set(CHRONOTOPE_CORE_LIB chronotope)
endif()
if(CHRONOTOPE_EDGELLM_BUILD)
  get_filename_component(EDGELLM_SRC "${CHRONOTOPE_EDGELLM_BUILD}/.." ABSOLUTE)
  set(TRT_INC "${CHRONOTOPE_TRT_PREFIX}/usr/include/aarch64-linux-gnu")
  set(TRT_LIB "${CHRONOTOPE_TRT_PREFIX}/usr/lib/aarch64-linux-gnu")
  if(NOT EXISTS "${CHRONOTOPE_EDGELLM_BUILD}/cpp/libedgellmCore.a" OR NOT EXISTS "${TRT_INC}/NvInfer.h")
    message(FATAL_ERROR "CHRONOTOPE_EDGELLM_BUILD/CHRONOTOPE_TRT_PREFIX: libedgellmCore.a or NvInfer.h not found")
  endif()
  target_compile_definitions(chronotope_objs PRIVATE CHRONOTOPE_WITH_EDGELLM CHRONOTOPE_EDGELLM_PLUGIN="${CHRONOTOPE_EDGELLM_BUILD}/libNvInfer_edgellm_plugin.so")
  target_include_directories(chronotope_objs PRIVATE ${EDGELLM_SRC}/cpp ${EDGELLM_SRC}/examples/multimodal ${EDGELLM_SRC}/examples/utils
                             ${EDGELLM_SRC}/3rdParty/nlohmannJson/include ${EDGELLM_SRC}/3rdParty/stb ${EDGELLM_SRC}/3rdParty/miniaudio)
  target_include_directories(chronotope_objs SYSTEM PRIVATE ${TRT_INC} ${CHRONOTOPE_CUDA_ARCH_DIR}/include)
  target_compile_options(chronotope_objs PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>)
  set(CHRONOTOPE_EDGELLM_LIBS
      ${CHRONOTOPE_EDGELLM_BUILD}/cpp/libedgellmCore.a dl
      ${EDGELLM_SRC}/cpp/kernels/cuteDSLArtifact/aarch64/sm_121/libcutedsl_aarch64.a cuda
      ${TRT_LIB}/libnvonnxparser.so ${TRT_LIB}/libnvinfer.so ${CHRONOTOPE_CUDA_ARCH_DIR}/lib/libcudart.so cudadevrt rt pthread dl)
  foreach(t chronotope_static chronotope)
    if(TARGET ${t})
      target_link_directories(${t} INTERFACE ${CHRONOTOPE_CUDA_ARCH_DIR}/lib ${CHRONOTOPE_CUDA_ARCH_DIR}/lib/stubs)
      target_link_options(${t} INTERFACE "-Wl,--unresolved-symbols=ignore-in-shared-libs" "-Wl,-rpath,${TRT_LIB}:${CHRONOTOPE_CUDA_ARCH_DIR}/lib")
      target_link_libraries(${t} INTERFACE ${CHRONOTOPE_EDGELLM_LIBS})
    endif()
  endforeach()
  if(TARGET chronotope)   # the .so device-links the Edge-LLM objects itself (as the binary does)
    target_link_directories(chronotope PRIVATE ${CHRONOTOPE_CUDA_ARCH_DIR}/lib ${CHRONOTOPE_CUDA_ARCH_DIR}/lib/stubs)
    target_link_options(chronotope PRIVATE "-Wl,--unresolved-symbols=ignore-in-shared-libs" "-Wl,-rpath,${TRT_LIB}:${CHRONOTOPE_CUDA_ARCH_DIR}/lib")
    target_link_libraries(chronotope PRIVATE ${CHRONOTOPE_EDGELLM_LIBS})
    set_target_properties(chronotope PROPERTIES CUDA_ARCHITECTURES 121a CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  endif()
  set_target_properties(chronotope_objs PROPERTIES CUDA_ARCHITECTURES 121a)
endif()

add_library(chronotope_synthetic STATIC src/synthetic.cpp)
target_include_directories(chronotope_synthetic PUBLIC src)

if(CHRONOTOPE_EDGELLM_BUILD)
  add_executable(chronotope_tests tests/test_main.cpp src/chronotope_devlink_stub.cu)   # device-links the Edge-LLM objects of the archive
  set_target_properties(chronotope_tests PROPERTIES CUDA_ARCHITECTURES 121a CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
else()
  add_executable(chronotope_tests tests/test_main.cpp)
endif()
target_link_libraries(chronotope_tests PRIVATE ${CHRONOTOPE_CORE_LIB} chronotope_synthetic pthread)

# ---- the in-process planner: TensorRT Edge-LLM (cards/T1-tensorrt-edge-llm-planner.md)
#   -DKGUA_EDGELLM_BUILD=<TensorRT-Edge-LLM/build>   (the runtime built for gb10; its source tree is the parent dir)
#   -DKGUA_TRT_PREFIX=<prefix holding usr/include/aarch64-linux-gnu/NvInfer.h and usr/lib/aarch64-linux-gnu/libnvinfer.so>

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu)
  find_package(ZLIB REQUIRED)
  find_package(Threads REQUIRED)
  # main.cu is the design's file name; it has no device code and nvcc's host front end fails on it, so g++ compiles it
  set_source_files_properties(src/main.cu PROPERTIES LANGUAGE CXX)
  set(CHRONOTOPE_MAIN_EXTRA src/chronotope_devlink_stub.cu src/planner_vlm.cpp src/planner_trt.cpp)
  if(CHRONOTOPE_EDGELLM_BUILD)
    set(CHRONOTOPE_MAIN_EXTRA src/chronotope_devlink_stub.cu)   # the planners live in the core library
  endif()
  add_executable(chronotope_main src/main.cu ${CHRONOTOPE_MAIN_EXTRA} src/capture.cpp src/kmbox.cpp src/trace.cpp src/serve.cpp)
  target_link_libraries(chronotope_main PRIVATE ${CHRONOTOPE_CORE_LIB} chronotope_synthetic ZLIB::ZLIB Threads::Threads)
  if(CHRONOTOPE_HOSTED_API)
    find_package(CURL REQUIRED)
    find_path(CHRONOTOPE_JSON_INCLUDE nlohmann/json.hpp REQUIRED)
    target_sources(chronotope_main PRIVATE src/planner_api.cpp)
    target_compile_definitions(chronotope_main PRIVATE CHRONOTOPE_HOSTED_API)
    target_include_directories(chronotope_main PRIVATE ${CHRONOTOPE_JSON_INCLUDE} ${CHRONOTOPE_CUDA_ARCH_DIR}/include)
    target_link_libraries(chronotope_main PRIVATE CURL::libcurl)
    add_executable(hosted_api_contract tests/test_hosted_api.cpp src/planner_api.cpp src/planner_vlm.cpp)
    target_include_directories(hosted_api_contract PRIVATE src ${CHRONOTOPE_JSON_INCLUDE})
    target_link_libraries(hosted_api_contract PRIVATE CURL::libcurl ZLIB::ZLIB Threads::Threads)
  endif()
  set_target_properties(chronotope_main PROPERTIES OUTPUT_NAME chronotope)
  if(CHRONOTOPE_EDGELLM_BUILD)
    get_filename_component(EDGELLM_SRC "${CHRONOTOPE_EDGELLM_BUILD}/.." ABSOLUTE)
    if(NOT EXISTS "${CHRONOTOPE_EDGELLM_BUILD}/cpp/libedgellmCore.a")
      message(FATAL_ERROR "CHRONOTOPE_EDGELLM_BUILD=${CHRONOTOPE_EDGELLM_BUILD}: cpp/libedgellmCore.a not found (build TensorRT-Edge-LLM first)")
    endif()
    set(TRT_INC "${CHRONOTOPE_TRT_PREFIX}/usr/include/aarch64-linux-gnu")
    set(TRT_LIB "${CHRONOTOPE_TRT_PREFIX}/usr/lib/aarch64-linux-gnu")
    if(NOT EXISTS "${TRT_INC}/NvInfer.h")
      message(FATAL_ERROR "CHRONOTOPE_TRT_PREFIX=${CHRONOTOPE_TRT_PREFIX}: usr/include/aarch64-linux-gnu/NvInfer.h not found")
    endif()
    # the edgellmCore objects are compiled for sm_121a with relocatable device code: the executable device-links them
    set_target_properties(chronotope_main PROPERTIES CUDA_ARCHITECTURES 121a CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_compile_definitions(chronotope_main PRIVATE CHRONOTOPE_WITH_EDGELLM CHRONOTOPE_EDGELLM_PLUGIN="${CHRONOTOPE_EDGELLM_BUILD}/libNvInfer_edgellm_plugin.so")
    target_include_directories(chronotope_main PRIVATE ${EDGELLM_SRC}/cpp ${EDGELLM_SRC}/examples/multimodal ${EDGELLM_SRC}/examples/utils
                               ${EDGELLM_SRC}/3rdParty/nlohmannJson/include ${EDGELLM_SRC}/3rdParty/stb ${EDGELLM_SRC}/3rdParty/miniaudio)
    target_include_directories(chronotope_main SYSTEM PRIVATE ${TRT_INC} ${CHRONOTOPE_CUDA_ARCH_DIR}/include)
    target_compile_options(chronotope_main PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>)
    # the link line of Edge-LLM's own llm_inference (build/examples/llm/CMakeFiles/llm_inference.dir/link.txt), made absolute
    target_link_directories(chronotope_main PRIVATE ${CHRONOTOPE_CUDA_ARCH_DIR}/lib ${CHRONOTOPE_CUDA_ARCH_DIR}/lib/stubs)
    target_link_options(chronotope_main PRIVATE "-Wl,--unresolved-symbols=ignore-in-shared-libs" "-Wl,-rpath,${TRT_LIB}:${CHRONOTOPE_CUDA_ARCH_DIR}/lib")
    target_link_libraries(chronotope_main PRIVATE
      ${CHRONOTOPE_EDGELLM_BUILD}/cpp/libedgellmCore.a
      dl
      ${EDGELLM_SRC}/cpp/kernels/cuteDSLArtifact/aarch64/sm_121/libcutedsl_aarch64.a
      cuda
      ${TRT_LIB}/libnvonnxparser.so ${TRT_LIB}/libnvinfer.so
      ${CHRONOTOPE_CUDA_ARCH_DIR}/lib/libcudart.so cudadevrt rt pthread dl)
    message(STATUS "chronotope: --planner trt enabled (Edge-LLM at ${EDGELLM_SRC}, TensorRT at ${CHRONOTOPE_TRT_PREFIX})")
  endif()
endif()
