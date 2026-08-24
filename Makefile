CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pedantic
LDFLAGS = -lm

SRCS = main.cpp tensorial_model.cpp mc_tensorial_model.cpp edmd_tensorial_model.cpp \
       extremal_tensorial_model.cpp
HEADERS = tensorial_model.h mc_tensorial_model.h edmd_tensorial_model.h \
          extremal_tensorial_model.h
OBJS = $(SRCS:.cpp=.o)
TARGET = tensorial_model

.PHONY: all clean depend

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) *~ core

depend: $(SRCS)
	makedepend -Y -- $(CXXFLAGS) -- $^

# For debugging
debug: CXXFLAGS += -g -O0
debug: clean all

# For profiling
profile: CXXFLAGS += -pg
profile: clean all
