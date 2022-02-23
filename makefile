CC = g++
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -Wall -funroll-loops -Wno-unused-result -g

all: word2vec word2vec_dbg

word2vec : src/word2vec.c
	$(CC) src/word2vec.c -o word2vec $(CFLAGS)

clean:
	rm -rf word2vec

