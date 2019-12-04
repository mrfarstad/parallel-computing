
#CC=gcc
CC=/usr/local/opt/llvm/bin/clang

FLAGS := -O3
LINKING := -fopenmp

main: main.c
	$(CC) $(FLAGS) $< -o $@ $(LINKING)

clean:
	rm -f main
