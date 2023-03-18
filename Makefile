CXXFLAGS = -DDEBUG -g -std=c++14 -I. --compiler-options -Wall
EXECUTABLES = bitonic odd-even batcher dsample samplesort

all: $(EXECUTABLES)

bitonic: bitonic.o common.o
	nvcc -O3 -o $@ $^

odd-even: odd-even.o common.o
	nvcc -O3 -o $@ $^

batcher: batcher.o common.o
	nvcc -O3 -o $@ $^

dsample: dsample.o common.o
	nvcc -O3 -o $@ $^

samplesort: samplesort.o common.o
	nvcc -O3 -o $@ $^

bitonic.o: bitonic.cu common.cuh
	nvcc $(CXXFLAGS) -o $@ -c $<

odd-even.o: odd-even.cu common.cuh
	nvcc $(CXXFLAGS) -o $@ -c $<

batcher.o: batcher.cu common.cuh
	nvcc $(CXXFLAGS) -o $@ -c $<

dsample.o: dsample.cu common.cuh
	nvcc $(CXXFLAGS) -o $@ -c $<

samplesort.o: samplesort.cu common.cuh
	nvcc $(CXXFLAGS) -o $@ -c $<

common.o: common.cu common.cuh
	nvcc $(CXXFLAGS) -o $@ -c $<

.PHONY: clean

clean:
	rm -f $(EXECUTABLES) *.o