CXX = nvcc
DEBUG_CXXFLAGS = -DDEBUG -g -std=c++14 --compiler-options -Wall
RELEASE_CXXFLAGS = -O3 -std=c++14 --compiler-options -Wall
DEBUG = bitonic-debug odd-even-debug
RELEASE = bitonic-release odd-even-release

debug: $(DEBUG)

release: $(RELEASE)

bitonic-debug: bitonic.cu common-debug.o
	$(CXX) $(DEBUG_CXXFLAGS) -o $@ $^

bitonic-release: bitonic.cu common-release.o
	$(CXX) $(RELEASE_CXXFLAGS) -o $@ $^

odd-even-debug: odd-even.cu common-debug.o
	$(CXX) $(DEBUG_CXXFLAGS) -o $@ $^

odd-even-release: odd-even.cu common-release.o
	$(CXX) $(RELEASE_CXXFLAGS) -o $@ $^

common-debug.o: common.cu common.cuh
	$(CXX) $(DEBUG_CXXFLAGS) -o $@ -c $<

common-release.o: common.cu common.cuh
	$(CXX) $(RELEASE_CXXFLAGS) -o $@ -c $<

.PHONY: clean

clean:
	rm -f $(DEBUG) $(RELEASE) *.o