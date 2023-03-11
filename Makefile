CXXFLAGS = -DDEBUG -dc -O3 -std=c++14 -I. --compiler-options -Wall
EXECUTABLES = bitonic odd-even

all: $(EXECUTABLES)

bitonic: bitonic.o common.o
	nvcc -O3 -o $@ $^

odd-even: odd-even.o common.o
	nvcc -O3 -o $@ $^

bitonic.o: bitonic.cu common.cuh
	nvcc $(CXXFLAGS) -o $@ -c $<

odd-even.o: odd-even.cu common.cuh
	nvcc $(CXXFLAGS) -o $@ -c $<

common.o: common.cu common.cuh
	nvcc $(CXXFLAGS) -o $@ -c $<

.PHONY: clean

clean:
	rm -f $(EXECUTABLES) *.o