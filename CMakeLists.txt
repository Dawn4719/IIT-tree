cmake_minimum_required(VERSION 3.16)
project(tree)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-g -O3 -march=native -Wall -mavx -pthread -lrocksdb)

FIND_PACKAGE(OpenMP REQUIRED)

#INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
#ADD_SUBDIRECTORY(cycle)
#ADD_SUBDIRECTORY(toolset)
#ADD_SUBDIRECTORY(util)

add_executable(bit_test
        graph/graph.h
        graph/graph.cpp
        utils/types.h
        utils/utils.h
        utils/CLI11.hpp
        utils/globals.h
        utils/globals.cpp
        matching/matching.h
        matching/main.cpp
        matching/matching.cpp
        matching/deltagraph.h
        matching/deltagraph.cpp
        matching/TiLa.h
        matching/TiLa.cpp
        matching/ticktock.h
        utils/pod.h
        roaring/roaring.hh
        roaring/roaring.h
        roaring/roaring.c
        matching/staticcore.hpp
        matching/reach.hpp
)

add_executable(RR
        graph/graph.h
        graph/graph.cpp
        utils/types.h
        utils/utils.h
        utils/CLI11.hpp
        utils/globals.h
        utils/globals.cpp
        matching/matching.h
        matching/mainRoaring.cpp
        matching/matching.cpp
        matching/deltagraph.h
        matching/deltagraph.cpp
        matching/TiLa.h
        matching/TiLa.cpp
        matching/ticktock.h
        utils/pod.h
        roaring/roaring.hh
        roaring/roaring.h
        roaring/roaring.c
        matching/staticcore.hpp
        matching/reach.hpp
)

find_library(ROCKSDB_LIB rocksdb)
find_package(Threads)
find_path(ROCKSDB_INCLUDE_DIR rocksdb/db.h
        PATHS "")
include_directories(${ROCKSDB_INCLUDE_DIR})

target_link_libraries(bit_test Threads::Threads ${ROCKSDB_LIB})
