
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(deep-imaging)

enable_language(CXX)
set (CMAKE_CXX_STANDARD 11)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

find_package(CUDA QUIET REQUIRED)

# locate tf
execute_process(
   COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
   OUTPUT_VARIABLE TF_INCLUDE
)
execute_process(
   COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
   OUTPUT_VARIABLE TF_LIB
)
string(STRIP ${TF_INCLUDE} TF_INCLUDE)
string(STRIP ${TF_LIB} TF_LIB)
message(STATUS "Tensorflow include path located at ${TF_INCLUDE}")
message(STATUS "Tensorflow lib located at ${TF_LIB}")

# build user-ops
add_subdirectory("user-ops")

# install python part
install(DIRECTORY tfcone/ DESTINATION tfcone FILES_MATCHING PATTERN "*.py")
