CXX = nvcc
DEBUG_CXXFLAGS = -DDEBUG -g -std=c++14 --compiler-options -Wall
RELEASE_CXXFLAGS = -O3 -std=c++14 --compiler-options -Wall
DEBUG = bitonic-debug
RELEASE = bitonic-release

debug: $(DEBUG)

release: $(RELEASE)

bitonic-debug: bitonic-debug.o common-debug.o
	$(CXX) $(DEBUG_CXXFLAGS) -o $@ $^

bitonic-release: bitonic-release.o common-release.o
	$(CXX) $(RELEASE_CXXFLAGS) -o $@ $^

bitonic-debug.o: bitonic.cu common.cuh
	$(CXX) -dc $(DEBUG_CXXFLAGS) -o $@ -c $<

bitonic-release.o: bitonic.cu common.cuh
	$(CXX) -dc $(RELEASE_CXXFLAGS) -o $@ -c $<

common-debug.o: common.cu common.cuh
	$(CXX) -dc $(DEBUG_CXXFLAGS) -o $@ -c $<

common-release.o: common.cu common.cuh
	$(CXX) -dc $(RELEASE_CXXFLAGS) -o $@ -c $<

.PHONY: clean

clean:
	rm -f $(DEBUG) $(RELEASE) *.o