CXX ?= g++
CXXFLAGS ?= -O2 -g -Wall -Wextra -Wshadow -Wno-missing-field-initializers -std=c++11 -I../src

all: skinned_simplify_test

skinned_simplify_test: skinned_simplify_test.cpp skinned_meshes.h ../src/simplifier.cpp
	$(CXX) skinned_simplify_test.cpp ../src/simplifier.cpp $(CXXFLAGS) -o $@

dump_lod: dump_lod.cpp skinned_meshes.h ../src/simplifier.cpp
	$(CXX) dump_lod.cpp ../src/simplifier.cpp $(CXXFLAGS) -o $@

run: skinned_simplify_test
	./skinned_simplify_test

images: dump_lod
	mkdir -p results
	./dump_lod
	python3 visualize_replay.py

clean:
	rm -f skinned_simplify_test dump_lod
