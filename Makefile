CXXFLAGS = -DDEBUG -dc -O3 -std=c++14 -I. --compiler-options -Wall
EXECUTABLES = bitonic odd-even batcher dsample rsample mergesort

all: $(EXECUTABLES)

bitonic: bitonic.o common.o
	nvcc -O3 -o $@ $^

odd-even: odd-even.o common.o
	nvcc -O3 -o $@ $^

batcher: batcher.o common.o
	nvcc -O3 -o $@ $^

dsample: dsample.o common.o
	nvcc -O3 -o $@ $^

rsample: rsample.o common.o
	nvcc -O3 -o $@ $^

mergesort: mergesort.o common.o
	nvcc -O3 -o $@ $^

bitonic.o: bitonic.cu common.cuh
	nvcc $(CXXFLAGS) -o $@ -c $<

odd-even.o: odd-even.cu common.cuh
	nvcc $(CXXFLAGS) -o $@ -c $<

batcher.o: batcher.cu common.cuh
	nvcc $(CXXFLAGS) -o $@ -c $<

dsample.o: dsample.cu common.cuh
	nvcc $(CXXFLAGS) -o $@ -c $<

rsample.o: rsample.cu common.cuh
	nvcc $(CXXFLAGS) -o $@ -c $<

mergesort.o: mergesort.cu common.cuh
	nvcc $(CXXFLAGS) -o $@ -c $<

common.o: common.cu common.cuh
	nvcc $(CXXFLAGS) -o $@ -c $<

.PHONY: clean

clean:
	rm -f $(EXECUTABLES) *.o
