# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# cmake-hgcc has no release tag yet. Pin the initial v1.0.0 commit so native
# builds do not change when its master branch advances.
set(VLLM_SAIL_CMAKE_HGCC_GIT_REPOSITORY
    "https://github.com/t-head/cmake-hgcc.git")
set(VLLM_SAIL_CMAKE_HGCC_GIT_TAG
    "5ca64a13a1487a2ba6a4ca7972d1cb73b2b56303")

function(_vllm_sail_find_hg_modules result)
  set(found "")
  foreach(dir IN LISTS CMAKE_MODULE_PATH)
    if(EXISTS "${dir}/CMakeDetermineHGCompiler.cmake")
      set(found "${dir}")
      break()
    endif()
  endforeach()
  set(${result} "${found}" PARENT_SCOPE)
endfunction()

function(vllm_sail_resolve_cmake_hgcc)
  set(cmake_hgcc_dir "")
  set(cmake_hgcc_source "")

  if(DEFINED CMAKE_HGCC_DIR AND NOT "${CMAKE_HGCC_DIR}" STREQUAL "")
    set(cmake_hgcc_dir "${CMAKE_HGCC_DIR}")
    set(cmake_hgcc_source "explicit")
  elseif(DEFINED ENV{CMAKE_HGCC_DIR} AND
         NOT "$ENV{CMAKE_HGCC_DIR}" STREQUAL "")
    set(cmake_hgcc_dir "$ENV{CMAKE_HGCC_DIR}")
    set(cmake_hgcc_source "explicit")
  endif()

  if(cmake_hgcc_dir)
    list(PREPEND CMAKE_MODULE_PATH
      "${cmake_hgcc_dir}/cmake/Modules"
      "${cmake_hgcc_dir}/cmake")
    list(PREPEND CMAKE_PREFIX_PATH "${cmake_hgcc_dir}")
  endif()
  if(DEFINED ENV{CMAKE_MODULE_PATH} AND
     NOT "$ENV{CMAKE_MODULE_PATH}" STREQUAL "")
    list(APPEND CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
  endif()

  _vllm_sail_find_hg_modules(hg_module_dir)
  if(cmake_hgcc_source STREQUAL "explicit" AND NOT hg_module_dir)
    message(FATAL_ERROR
      "CMAKE_HGCC_DIR=${cmake_hgcc_dir} does not contain cmake-hgcc's "
      "cmake/Modules/CMakeDetermineHGCompiler.cmake")
  endif()

  # An installed cmake-hgcc config package prepends its module directory. This
  # also honours CMAKE_PREFIX_PATH without requiring a checkout-specific path.
  if(NOT hg_module_dir)
    find_package(cmake-hgcc CONFIG QUIET)
    _vllm_sail_find_hg_modules(hg_module_dir)
    if(hg_module_dir)
      set(cmake_hgcc_source "installed")
    endif()
  endif()

  if(NOT hg_module_dir)
    include(FetchContent)
    message(STATUS
      "vllm-sail: cmake-hgcc is not configured; fetching "
      "${VLLM_SAIL_CMAKE_HGCC_GIT_REPOSITORY} at "
      "${VLLM_SAIL_CMAKE_HGCC_GIT_TAG}")
    FetchContent_Declare(cmake_hgcc
      GIT_REPOSITORY "${VLLM_SAIL_CMAKE_HGCC_GIT_REPOSITORY}"
      GIT_TAG "${VLLM_SAIL_CMAKE_HGCC_GIT_TAG}"
      GIT_SHALLOW FALSE
      GIT_PROGRESS TRUE
      GIT_SUBMODULES ""
      # Fetch the sources without adding cmake-hgcc's install-only project.
      SOURCE_SUBDIR "_vllm_sail_fetch_only")
    FetchContent_MakeAvailable(cmake_hgcc)
    set(cmake_hgcc_dir "${cmake_hgcc_SOURCE_DIR}")
    list(PREPEND CMAKE_MODULE_PATH "${cmake_hgcc_dir}/cmake/Modules")
    list(PREPEND CMAKE_PREFIX_PATH "${cmake_hgcc_dir}")
    set(cmake_hgcc_source "fetched")
    _vllm_sail_find_hg_modules(hg_module_dir)
  elseif(NOT cmake_hgcc_source)
    set(cmake_hgcc_source "configured")
  endif()

  if(NOT hg_module_dir)
    message(FATAL_ERROR
      "cmake-hgcc's CMake language modules were not found after discovery "
      "(CMAKE_MODULE_PATH=${CMAKE_MODULE_PATH})")
  endif()

  if(cmake_hgcc_dir)
    set(CMAKE_HGCC_DIR "${cmake_hgcc_dir}" PARENT_SCOPE)
  endif()
  set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)
  set(VLLM_SAIL_CMAKE_HGCC_SOURCE "${cmake_hgcc_source}" PARENT_SCOPE)
  message(STATUS
    "vllm-sail: using cmake-hgcc modules from ${hg_module_dir} "
    "(${cmake_hgcc_source})")
endfunction()
