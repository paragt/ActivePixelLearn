# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /misc/local/cmake-2.8.8/bin/cmake

# The command to remove a file.
RM = /misc/local/cmake-2.8.8/bin/cmake -E remove -f

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /misc/local/cmake-2.8.8/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised

# Include any dependencies generated for this target.
include CMakeFiles/SemiSupervised.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/SemiSupervised.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SemiSupervised.dir/flags.make

CMakeFiles/SemiSupervised.dir/kmeans.cpp.o: CMakeFiles/SemiSupervised.dir/flags.make
CMakeFiles/SemiSupervised.dir/kmeans.cpp.o: kmeans.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/SemiSupervised.dir/kmeans.cpp.o"
	/usr/local/gcc/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/SemiSupervised.dir/kmeans.cpp.o -c /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised/kmeans.cpp

CMakeFiles/SemiSupervised.dir/kmeans.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SemiSupervised.dir/kmeans.cpp.i"
	/usr/local/gcc/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised/kmeans.cpp > CMakeFiles/SemiSupervised.dir/kmeans.cpp.i

CMakeFiles/SemiSupervised.dir/kmeans.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SemiSupervised.dir/kmeans.cpp.s"
	/usr/local/gcc/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised/kmeans.cpp -o CMakeFiles/SemiSupervised.dir/kmeans.cpp.s

CMakeFiles/SemiSupervised.dir/kmeans.cpp.o.requires:
.PHONY : CMakeFiles/SemiSupervised.dir/kmeans.cpp.o.requires

CMakeFiles/SemiSupervised.dir/kmeans.cpp.o.provides: CMakeFiles/SemiSupervised.dir/kmeans.cpp.o.requires
	$(MAKE) -f CMakeFiles/SemiSupervised.dir/build.make CMakeFiles/SemiSupervised.dir/kmeans.cpp.o.provides.build
.PHONY : CMakeFiles/SemiSupervised.dir/kmeans.cpp.o.provides

CMakeFiles/SemiSupervised.dir/kmeans.cpp.o.provides.build: CMakeFiles/SemiSupervised.dir/kmeans.cpp.o

CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.o: CMakeFiles/SemiSupervised.dir/flags.make
CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.o: weightmatrix1.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.o"
	/usr/local/gcc/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.o -c /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised/weightmatrix1.cpp

CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.i"
	/usr/local/gcc/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised/weightmatrix1.cpp > CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.i

CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.s"
	/usr/local/gcc/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised/weightmatrix1.cpp -o CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.s

CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.o.requires:
.PHONY : CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.o.requires

CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.o.provides: CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.o.requires
	$(MAKE) -f CMakeFiles/SemiSupervised.dir/build.make CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.o.provides.build
.PHONY : CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.o.provides

CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.o.provides.build: CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.o

CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.o: CMakeFiles/SemiSupervised.dir/flags.make
CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.o: weightmatrix2.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.o"
	/usr/local/gcc/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.o -c /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised/weightmatrix2.cpp

CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.i"
	/usr/local/gcc/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised/weightmatrix2.cpp > CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.i

CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.s"
	/usr/local/gcc/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised/weightmatrix2.cpp -o CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.s

CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.o.requires:
.PHONY : CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.o.requires

CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.o.provides: CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.o.requires
	$(MAKE) -f CMakeFiles/SemiSupervised.dir/build.make CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.o.provides.build
.PHONY : CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.o.provides

CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.o.provides.build: CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.o

# Object files for target SemiSupervised
SemiSupervised_OBJECTS = \
"CMakeFiles/SemiSupervised.dir/kmeans.cpp.o" \
"CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.o" \
"CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.o"

# External object files for target SemiSupervised
SemiSupervised_EXTERNAL_OBJECTS =

libSemiSupervised.so: CMakeFiles/SemiSupervised.dir/kmeans.cpp.o
libSemiSupervised.so: CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.o
libSemiSupervised.so: CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.o
libSemiSupervised.so: CMakeFiles/SemiSupervised.dir/build.make
libSemiSupervised.so: /groups/flyem/proj/cluster/toufiq/SuiteSparse/sparse-install/lib/libcholmod.a
libSemiSupervised.so: /groups/flyem/proj/cluster/toufiq/SuiteSparse/sparse-install/lib/libcolamd.a
libSemiSupervised.so: /groups/flyem/proj/cluster/toufiq/SuiteSparse/sparse-install/lib/libsuitesparseconfig.a
libSemiSupervised.so: /groups/flyem/proj/cluster/toufiq/SuiteSparse/sparse-install/lib/libamd.a
libSemiSupervised.so: /groups/flyem/proj/cluster/toufiq/SuiteSparse/sparse-install/lib/libbtf.a
libSemiSupervised.so: /groups/flyem/proj/cluster/toufiq/SuiteSparse/sparse-install/lib/libccolamd.a
libSemiSupervised.so: /groups/flyem/proj/cluster/toufiq/SuiteSparse/sparse-install/lib/libcamd.a
libSemiSupervised.so: /groups/flyem/proj/cluster/toufiq/SuiteSparse/sparse-install/lib/libklu.a
libSemiSupervised.so: /groups/flyem/proj/cluster/toufiq/SuiteSparse/sparse-install/lib/libldl.a
libSemiSupervised.so: /groups/flyem/proj/cluster/toufiq/SuiteSparse/sparse-install/lib/librbio.a
libSemiSupervised.so: /groups/flyem/proj/cluster/toufiq/SuiteSparse/sparse-install/lib/libspqr.a
libSemiSupervised.so: /groups/flyem/proj/cluster/toufiq/SuiteSparse/sparse-install/lib/libumfpack.a
libSemiSupervised.so: /groups/flyem/proj/cluster/toufiq/SuiteSparse/sparse-install/lib/libmetis.a
libSemiSupervised.so: /usr/lib64/liblapack.so.3
libSemiSupervised.so: /usr/lib64/libblas.so.3
libSemiSupervised.so: /usr/lib64/libgfortran.so.3
libSemiSupervised.so: /usr/lib64/libm.so
libSemiSupervised.so: /usr/lib64/librt.so
libSemiSupervised.so: CMakeFiles/SemiSupervised.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library libSemiSupervised.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SemiSupervised.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SemiSupervised.dir/build: libSemiSupervised.so
.PHONY : CMakeFiles/SemiSupervised.dir/build

CMakeFiles/SemiSupervised.dir/requires: CMakeFiles/SemiSupervised.dir/kmeans.cpp.o.requires
CMakeFiles/SemiSupervised.dir/requires: CMakeFiles/SemiSupervised.dir/weightmatrix1.cpp.o.requires
CMakeFiles/SemiSupervised.dir/requires: CMakeFiles/SemiSupervised.dir/weightmatrix2.cpp.o.requires
.PHONY : CMakeFiles/SemiSupervised.dir/requires

CMakeFiles/SemiSupervised.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SemiSupervised.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SemiSupervised.dir/clean

CMakeFiles/SemiSupervised.dir/depend:
	cd /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised /groups/flyem/proj/cluster/toufiq/goo/SemiSupervised/CMakeFiles/SemiSupervised.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/SemiSupervised.dir/depend
