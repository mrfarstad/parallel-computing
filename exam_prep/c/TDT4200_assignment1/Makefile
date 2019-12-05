CC=gcc
DEPS=bitmap.h
OBJ=bitmap.o main.o

.PHONY: main run open

%.o: %.c %(DEPS)
	$(CC) -c -o %@

main: $(OBJ)
	$(CC) -o $@ $^

run:
	./main && open after.bmp

open: main run
	open after.bmp

clean:
	rm *.o main
