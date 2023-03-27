message(STATUS "RapidJSON not found. Downloading and installing it...")

execute_process(COMMAND sudo apt-get install -y rapidjson-dev
  RESULT_VARIABLE result)

if(result)
  include(FetchContent)
  FetchContent_Declare(rapidjson
    GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
    GIT_TAG v1.1.0
  )
  FetchContent_MakeAvailable(rapidjson)

  # set(RAPIDJSON_BUILD_TESTS OFF CACHE INTERNAL "")
  set(RapidJSON_INCLUDE_DIRS ${rapidjson_SOURCE_DIR}/include)
endif()