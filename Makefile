# Compiler and flags
CC = nvcc
CFLAGS += -std=c++14 -O3 -Xcompiler -Wall,-Wextra 
LDFLAGS = -lm
GL_LDFLAGS = -lGL -lglfw

# Files (all .cu now)
TARGETS = tiny_ising demo

# Rules
all: $(TARGETS)

tiny_ising: tiny_ising.cu ising.cu xoshiro256plus.cu wtime.cu
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

demo: demo.cu ising.cu xoshiro256plus.cu wtime.cu
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(GL_LDFLAGS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: clean all
