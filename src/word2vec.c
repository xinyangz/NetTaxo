// Adapted from Google word2vec.c code

//  Original Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define MAX_STRING 300
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define MAX_NODE_TYPE 10

const int vocab_hash_size =
    30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;  // Precision of float numbers

struct vocab_word {
  long long cn;
  char *word;
};

char train_file[MAX_STRING], output_file_base[MAX_STRING],
    output_file[MAX_STRING], train_file_net[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING], save_vocab_net_file[MAX_STRING];
// files for embedding loading
char syn0_base_file[MAX_STRING], syn1_base_file[MAX_STRING];
char syn0_file[MAX_STRING], syn1_file[MAX_STRING];
// files for embedding saving
char syn0_base_out_file[MAX_STRING], syn1_base_out_file[MAX_STRING];
char syn0_out_file[MAX_STRING], syn1_out_file[MAX_STRING];
char syn1_net_out_file[MAX_STRING];

struct vocab_word *vocab, *vocab_net;
int binary = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12,
    min_reduce = 1, pp = 0;
int *vocab_hash, *vocab_hash_net;
long long vocab_max_size = 1000, vocab_max_size_net = 1000, vocab_size = 0, vocab_size_net = 0, layer1_size = 100;
long long train_words = 0, train_words_net = 0, word_count_actual = 0, iter = 5,
          file_size = 0, file_size_net = 0;
real alpha = 0.025, starting_alpha, sample = 1e-4;
real lambda_base = 0.001, lambda_net = 0.5;
real *syn0, *syn1, *syn1_net, *expTable;
// fixed offset vectors, used for incremental learning
real *syn0_base, *syn1_base;
int incremental = 0;
clock_t start;

int negative = 5;
const int table_size = 1e8;
int *table, *table_net;
int net = 0;

// typed negative sampling
std::vector<long long> typed_table[MAX_NODE_TYPE];

// build typed tables for heterogeneous negative sampling
void NodeType() {
  long long i;
  long long target;
  int type_idx;
  for (i = 0; i < table_size; i++) {
    if (i % 1000 == 0) {
      printf("nodetype %d\n", i);
    }
    target = table_net[i];
    if (target == 0) continue;

    type_idx = strtol(vocab[target].word, NULL, 10);

    if (type_idx > MAX_NODE_TYPE) continue;
    typed_table[type_idx].push_back(target);
  }
}

void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

void InitNetUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table_net = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size_net; a++) train_words_pow += pow(vocab_net[a].cn, power);
  i = 0;
  d1 = pow(vocab_net[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table_net[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab_net[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size_net) i = vocab_size_net - 1;
  }
  if (pp) {
    NodeType();
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word
// boundaries
void ReadWord(char *word, FILE *fin, char *eof) {
  int a = 0, ch;
  while (1) {
    ch = getc_unlocked(fin);
    if (ch == EOF) {
      *eof = 1;
      break;
    }
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else
        continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;  // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found,
// returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

int SearchVocabNet(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash_net[hash] == -1) return -1;
    if (!strcmp(word, vocab_net[vocab_hash_net[hash]].word)) return vocab_hash_net[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin, char *eof) {
  char word[MAX_STRING], eof_l = 0;
  ReadWord(word, fin, &eof_l);
  if (eof_l) {
    *eof = 1;
    return -1;
  }
  return SearchVocab(word);
}

// Reads a node and returns its index in the node vocab
int ReadNodeIndex(FILE *fin, char *eof) {
  char word[MAX_STRING], eof_l = 0;
  ReadWord(word, fin, &eof_l);
  if (eof_l) {
    *eof = 1;
    return -1;
  }
  return SearchVocabNet(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(
        vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

int AddNodeToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab_net[vocab_size_net].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab_net[vocab_size_net].word, word);
  vocab_net[vocab_size_net].cn = 0;
  vocab_size_net++;
  // Reallocate memory if needed
  if (vocab_size_net + 2 >= vocab_max_size_net) {
    vocab_max_size_net += 1000;
    vocab_net = (struct vocab_word *)realloc(
        vocab_net, vocab_max_size_net * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash_net[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash_net[hash] = vocab_size_net - 1;
  return vocab_size_net - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  long long l = ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
  if (l > 0) return 1;
  if (l < 0) return -1;
  return 0;
}

int StrStartsWith(char *str, char *pre) {
  int len_pre = strlen(pre), len_str = strlen(str);
  if (len_str < len_pre) return 0;
  return strncmp(pre, str, len_pre) == 0;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn <= min_reduce) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash = GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(
      vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  printf("sort complete\n");
}

void SortVocabNet() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  printf("vocab_size_net %lld\n", vocab_size_net);
  qsort(&vocab_net[1], vocab_size_net - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash_net[a] = -1;
  size = vocab_size_net;
  train_words_net = 0;
  for (a = 0; a < size; a++) {
    // Hash will be re-computed, as after the sorting it is not actual
    hash = GetWordHash(vocab_net[a].word);
    while (vocab_hash_net[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash_net[hash] = a;
    train_words_net += vocab_net[a].cn;
  }
  vocab_net = (struct vocab_word *)realloc(
      vocab_net, (vocab_size_net + 1) * sizeof(struct vocab_word));
  printf("sort complete\n");
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++)
    if (vocab[a].cn > min_reduce || StrStartsWith(vocab[a].word, "<phrase>")) {
      vocab[b].cn = vocab[a].cn;
      vocab[b].word = vocab[a].word;
      b++;
    } else
      free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING], eof = 0;
  FILE *fin;
  long long a, i, wc = 0;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin, &eof);
    if (eof) break;
    train_words++;
    wc++;
    if ((debug_mode > 1) && (wc >= 1000000)) {
      printf("%lldM%c", train_words / 1000000, 13);
      fflush(stdout);
      wc = 0;
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      if (StrStartsWith(word, "<phrase>")) {
        vocab[a].cn = min_reduce + 1;
      } else {
        vocab[a].cn = 1;
      }
    } else
      vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}


void LearnNetVocab() {
  char word[MAX_STRING], eof = 0, line_start = 1;
  FILE *fin;
  long long a, i, wc = 0;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash_net[a] = -1;
  fin = fopen(train_file_net, "rb");
  if (fin == NULL) {
    printf("%s\n", train_file_net);
    printf("ERROR: net training data file not found!\n");
    exit(1);
  }
  vocab_size_net = 0;
  AddNodeToVocab((char *)"</s>");
  while (1) {
    if (line_start) {
      ReadWord(word, fin, &eof);
      line_start = 0;
      continue;
    }
    ReadWord(word, fin, &eof);
    if (!strcmp(word, "</s>"))
      line_start = 1;
    if (eof) break;
    train_words_net++;
    wc++;
    if ((debug_mode > 1) && (wc >= 1000000)) {
      printf("%lldM%c", train_words_net / 1000000, 13);
      fflush(stdout);
      wc = 0;
    }
    i = SearchVocabNet(word);
    if (i == -1) {
      a = AddNodeToVocab(word);
      vocab_net[a].cn = 1;
    } else
      vocab_net[i].cn++;
  }
  printf("Words in net file: %lld\n", train_words_net);
  SortVocabNet();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size_net);
    printf("Words in train file: %lld\n", train_words_net);
  }
  file_size_net = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++)
    fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c, eof = 0;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin, &eof);
    if (eof) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

// Allocate memory for input and output vectors.
// Random initialize them.
void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&syn0, 128,
                     (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {
    printf("Memory allocation failed\n");
    exit(1);
  }
  if (negative > 0) {
    a = posix_memalign((void **)&syn1, 128,
                       (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {
      printf("Memory allocation failed\n");
      exit(1);
    }
    for (a = 0; a < vocab_size; a++)
      for (b = 0; b < layer1_size; b++) syn1[a * layer1_size + b] = 0;
    if (net) {
      a = posix_memalign((void **)&syn1_net, 128,
                        (long long)vocab_size_net * layer1_size * sizeof(real));
      if (syn1_net == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
      }
      for (a = 0; a < vocab_size_net; a++)
        for (b = 0; b < layer1_size; b++) syn1_net[a * layer1_size + b] = 0;
    }
  }

  // intialize offset vectors for input and output vectors
  a = posix_memalign((void **)&syn0_base, 128,
                     (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0_base == NULL) {
    printf("Memory allocation failed\n");
    exit(1);
  }
  a = posix_memalign((void **)&syn1_base, 128,
                     (long long)vocab_size * layer1_size * sizeof(real));
  if (syn1_base == NULL) {
    printf("Memory allocation failed\n");
    exit(1);
  }
  for (a = 0; a < vocab_size; a++)
    for (b = 0; b < layer1_size; b++) {
      syn0_base[a * layer1_size + b] = 0;
      syn1_base[a * layer1_size + b] = 0;
    }
  for (a = 0; a < vocab_size; a++)
    for (b = 0; b < layer1_size; b++) {
      if (incremental > 0) {
        syn0[a * layer1_size + b] = 0;
      } else {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn0[a * layer1_size + b] =
            (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
      }
    }
}

void *TrainModelThread(void *id) {
  long long a, b, d, e, word, context_word_net, target_word_net, node, last_word,
      sentence_length = 0, sentence_position = 0, net_context_length = 0;
  long long word_count = 0, last_word_count = 0, word_count_net = 0,
            sen[MAX_SENTENCE_LENGTH + 1], context_net[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, context, label, local_iter = iter;
  int type_idx;
  unsigned long long next_random = (long long)id;
  char eof = 0, eof_net = 0;
  // stats collected from network training file
  long long line_count = 0, oov_count = 0;
  real f, g;
  clock_t now;
  // real *neu1 = (real *)calloc(layer1_size, sizeof(real));   // activation
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));  // loss
  FILE *fi = fopen(train_file, "rb");
  FILE *fi_net = fopen(train_file_net, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  if (net) {
    fseek(fi_net, file_size_net / (long long)num_threads * (long long)id, SEEK_SET);
    while (ReadWordIndex(fi_net, &eof_net) != 0);  // skip one line
  }

  while (1) {
    // This part adjusts alpha according to training progress on
    // text file. Network is not involved in this part.
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now = clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13,
               alpha, word_count_actual / (real)(iter * train_words + 1) * 100,
               word_count_actual /
                   ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha *
              (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    // read a single sentence into `sen`
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi, &eof);
        if (eof) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the
        // ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) *
                     (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (eof || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      eof = 0;
      continue;
    }
    // The following part predicts context based on a target word, and
    // backpropagates accordingly
    word = sen[sentence_position];  // context word
    if (word == -1) continue;
    // for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;  // shrink context window, but why??
    // train skip-gram
    for (a = b; a < window * 2 + 1 - b; a++)
      // a is the amount of offset in the window
      if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];  // target word
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX DELETED
        // NEGATIVE SAMPLING
        if (negative > 0)
          for (d = 0; d < negative + 1; d++) {
            if (d == 0) {
              target = word;
              label = 1;
            } else {
              next_random = next_random * (unsigned long long)25214903917 + 11;
              target = table[(next_random >> 16) % table_size];
              if (target == 0) target = next_random % (vocab_size - 1) + 1;
              if (target == word) continue;
              label = 0;
            }
            l2 = target * layer1_size;
            f = 0;
            // l1 - target word
            // l2 - negative sampled word or context word
            // f - syn0[l1] .* syn1[l2]
            // g - (t_i - sigmoid(f))
            for (c = 0; c < layer1_size; c++)
              f += (syn0_base[c + l1] + syn0[c + l1]) *
                   (syn1_base[c + l2] + syn1[c + l2]);
            if (f > MAX_EXP)
              g = (label - 1);
            else if (f < -MAX_EXP)
              g = (label - 0);
            else
              g = (label - expTable[(int)((f + MAX_EXP) *
                                          (EXP_TABLE_SIZE / MAX_EXP / 2))]);
            for (c = 0; c < layer1_size; c++)
              neu1e[c] += (1 - lambda_net) * alpha *
                          g * (syn1_base[c + l2] + syn1[c + l2]);
            for (c = 0; c < layer1_size; c++)
              syn1[c + l2] += (1 - lambda_net) * alpha * (g * (syn0_base[c + l1] + syn0[c + l1]) -
                                       lambda_base * syn1[c + l2]);
            // regularization
            // for (c = 0; c < layer1_size; c++)
            //   syn1[c + l2] -= alpha * lambda_base * syn1[c + l2];
        }
        // Learn weights input -> hidden
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c] - lambda_base * syn0[c + l1];
        // regularization
        // for (c = 0; c < layer1_size; c++)
        //   syn0[c + l1] -= alpha * lambda_base * syn0[c + l1];
      }
    // This part moves context word to next word in the sentence
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }

    if (net) for (e = 0; e < window; e++) {
      // now per each target -> context training in text, we read
      // one line from the network corpus file, and run target -> context
      // training once
      // Note that the target word read from the network corpus is different
      // from the current target word in text file.

      // try to read one line from network file
      line_count++;
      target_word_net = ReadWordIndex(fi_net, &eof_net);
      if (eof_net) {
        fseek(fi_net, file_size_net / (long long)num_threads * (long long)id, SEEK_SET);
        eof_net = 0;
        while (ReadWordIndex(fi_net, &eof_net) != 0);  // skip one line
        target_word_net = ReadWordIndex(fi_net, &eof_net);
        continue;
      }
      // oov word
      if (target_word_net == -1) {
        // skip this line
        oov_count++;
        while (ReadWordIndex(fi_net, &eof_net) != 0);
        continue;
      }
      // empty neighboring nodes
      if (target_word_net == 0) continue;
      net_context_length = 0;
      while (1) {
        node = ReadNodeIndex(fi_net, &eof_net);
        if (eof_net) break;
        if (node == -1) continue;
        word_count_net++;
        if (node == 0) break;
        if (sample > 0) {
          real ran = (sqrt(vocab_net[node].cn / (sample * train_words_net)) + 1) *
                    (sample * train_words_net) / vocab_net[node].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        context_net[net_context_length++] = node;
        if (net_context_length >= MAX_SENTENCE_LENGTH) break;
      }
      if (eof_net || (word_count_net > train_words_net / num_threads)) {
        word_count_net = 0;
        net_context_length = 0;
        fseek(fi_net, file_size_net / (long long)num_threads * (long long)id, SEEK_SET);
        eof_net = 0;
        while (ReadWordIndex(fi_net, &eof_net) != 0);  // skip one line
        continue;
      }
      // train one sentence
      // target word already set
      l1 = target_word_net * layer1_size;
      // for (c = 0; c < layer1_size; c++) neu1[c] = 0;
      for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
      for (a = 0; a < net_context_length; a++) {
        context_word_net = context_net[a];
        if (context_word_net == -1) continue;
        if (negative > 0) {
          for (d = 0; d < negative + 1; d++) {
            if (d == 0) {
              context = context_word_net;
              label = 1;
            } else {
              next_random = next_random * (unsigned long long)25214903917 + 11;

              if (pp) {
                type_idx = strtol(vocab[context_word_net].word, NULL, 10);
                if (type_idx >= MAX_NODE_TYPE) {
                  context = table_net[(next_random >> 16) % table_size];
                  if (context == 0) context = next_random % (vocab_size_net - 1) + 1;
                  if (context == context_word_net) continue;
                } else {
                  context = typed_table[type_idx][(next_random >> 16) % typed_table[type_idx].size()];
                  if (context == 0) context = next_random % (vocab_size_net - 1) + 1;
                }
              } else {
                context = table_net[(next_random >> 16) % table_size];
                if (context == 0) context = next_random % (vocab_size_net - 1) + 1;
                if (context == context_word_net) continue;
              }

              label = 0;
            }
            l2 = context * layer1_size;
            f = 0;
            for (c = 0; c < layer1_size; c++)
              f += (syn0_base[c + l1] + syn0[c + l1]) * syn1_net[c + l2];
            if (f > MAX_EXP)
                g = (label - 1);
            else if (f < -MAX_EXP)
              g = (label - 0);
            else
              g = (label - expTable[(int)((f + MAX_EXP) *
                                          (EXP_TABLE_SIZE / MAX_EXP / 2))]);
            for (c = 0; c < layer1_size; c++)
              neu1e[c] += lambda_net * alpha * g * syn1_net[c + l2];
            for (c = 0; c < layer1_size; c++)
              syn1_net[c + l2] += lambda_net * alpha * g * (syn0_base[c + l1] + syn0[c + l1]);
          }
        }
      }
      for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
    }
  }
  fclose(fi);
  if (net)
    fclose(fi_net);
  // free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

// Load word embedding from file. If file not found,
// load nothing and return.
// Args:
//    W: the embedding to be load
//    filename: the file to load embedding from
void LoadWordVector(real *W, char *filename) {
  char ch;
  char word[MAX_STRING];
  long long n_words, n_dim;
  int i, j, word_idx, tmp;
  FILE *fi;
  if (filename[0] == 0) return;
  printf("Loading word vector from %s\n", filename);
  fi = fopen(filename, "rb");
  if (fi == NULL) {
    fprintf(stderr, "Cannot open file %s\n", filename);
    return;
  }
  tmp = fscanf(fi, "%lld %lld", &n_words, &n_dim);
  if (tmp != 2) {
    fprintf(stderr, "Failed to parse first line of file %s\n", filename);
    fprintf(stderr, "Loaded nothing, proceeding...\n");
    return;
  }
  if (n_dim != layer1_size) {
    fprintf(stderr, "Dimensionality mismatch in %s, loaded nothing\n", filename);
    return;
  }
  assert(n_dim == layer1_size);
  for (i = 0; i < n_words; i++) {
    fscanf(fi, "%s%c", word, &ch);
    word_idx = SearchVocab(word);
    if (word_idx == -1) continue;
    if (binary > 0)
      for (j = 0; j < n_dim; j++)
        fread(&W[word_idx * layer1_size + j], sizeof(real), 1, fi);
    else
      for (j = 0; j < n_dim; j++)
        fscanf(fi, "%f", &W[word_idx * layer1_size + j]);
  }
}

// Save word embedding to file.
// Args:
//    W: The array of embedding vectors.
//    filename: The file to save word embeddings.
void SaveWordVector(real *W, char *filename) {
  long a, b;
  FILE *fo;
  fo = fopen(filename, "wb");
  // fo = fopen(output_file, "wb");
  if (fo == NULL) {
    fprintf(stderr, "Saving word vectors failed because cannot open file %s\n", filename);
    return;
  }
  fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
  for (a = 0; a < vocab_size; a++) {
    fprintf(fo, "%s ", vocab[a].word);
    if (binary)
      for (b = 0; b < layer1_size; b++)
        fwrite(&W[a * layer1_size + b], sizeof(real), 1, fo);
    else
      for (b = 0; b < layer1_size; b++)
        fprintf(fo, "%lf ", W[a * layer1_size + b]);
    fprintf(fo, "\n");
  }
  fclose(fo);
}

void SaveWordVectorNet(real *W, char *filename) {
  long a, b;
  FILE *fo;
  fo = fopen(filename, "wb");
  // fo = fopen(output_file, "wb");
  if (fo == NULL) {
    fprintf(stderr, "Saving word vectors failed because cannot open file %s\n", filename);
    return;
  }
  fprintf(fo, "%lld %lld\n", vocab_size_net, layer1_size);
  for (a = 0; a < vocab_size_net; a++) {
    fprintf(fo, "%s ", vocab_net[a].word);
    if (binary)
      for (b = 0; b < layer1_size; b++)
        fwrite(&W[a * layer1_size + b], sizeof(real), 1, fo);
    else
      for (b = 0; b < layer1_size; b++)
        fprintf(fo, "%lf ", W[a * layer1_size + b]);
    fprintf(fo, "\n");
  }
  fclose(fo);
}

void TrainModel() {
  long a;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  LearnVocabFromTrainFile();
  if (net)
    LearnNetVocab();
  if (save_vocab_file[0] != 0) SaveVocab();
  InitNet();
  printf("Loading word vectors\n");
  LoadWordVector(syn0, syn0_file);
  LoadWordVector(syn1, syn1_file);
  LoadWordVector(syn0_base, syn0_base_file);
  LoadWordVector(syn1_base, syn1_base_file);
  if (negative > 0)
    InitUnigramTable();
  if (net)
    InitNetUnigramTable();
  printf("Start training\n");
  start = clock();
  for (a = 0; a < num_threads; a++)
    pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

  // Save the word vectors
  printf("\nSaving word vectors\n");
  printf("Saving syn0 to %s\n", syn0_out_file);
  SaveWordVector(syn0, syn0_out_file);
  printf("Saving syn1 to %s\n", syn1_out_file);
  SaveWordVector(syn1, syn1_out_file);
  printf("Saving syn0_base to %s\n", syn0_base_out_file);
  SaveWordVector(syn0_base, syn0_base_out_file);
  printf("Saving syn1_base to %s\n", syn1_base_out_file);
  SaveWordVector(syn1_base, syn1_base_out_file);
  printf("Saving syn1_net to %s\n", syn1_net_out_file);
  SaveWordVectorNet(syn1_net, syn1_net_out_file);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        printf("Argument missing for %s\n", str);
        exit(1);
      }
      return a;
    }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf(
        "\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf(
        "\t\tSet threshold for occurrence of words. Those that appear with "
        "higher frequency in the training data\n");
    printf(
        "\t\twill be randomly down-sampled; default is 1e-3, useful range is "
        "(0, 1e-5)\n");
    printf("\t-negative <int>\n");
    printf(
        "\t\tNumber of negative examples; default is 5, common values are 3 - "
        "10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf(
        "\t\tThis will discard words that appear less than <int> times; "
        "default is 5\n");
    printf("\t-alpha <float>\n");
    printf(
        "\t\tSet the starting learning rate; default is 0.025 for skip-gram\n");
    printf("\t-debug <int>\n");
    printf(
        "\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf(
        "\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf(
        "\t\tThe vocabulary will be read from <file>, not constructed from the "
        "training data\n");
    printf("\nExamples:\n");
    printf(
        "./word2vec -train data.txt -output vec.txt -size 200 -window 5 "
        "-sample 1e-4 -negative 5 -binary 0 -iter 3\n\n");
    return 0;
  }
  output_file_base[0] = 0;
  save_vocab_file[0] = 0;
  save_vocab_net_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0)
    layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0)
    strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-train-net", argc, argv)) > 0)
    strcpy(train_file_net, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0)
    strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab-net", argc, argv)) > 0)
    strcpy(save_vocab_net_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0)
    strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0)
    debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0)
    binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0)
    strcpy(output_file_base, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0)
    window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0)
    sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-pp", argc, argv)) > 0)
    pp = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0)
    negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-net", argc, argv)) > 0)
    net = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0)
    num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0)
    min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-incremental", argc, argv)) > 0)
    incremental = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-syn0-base", argc, argv)) > 0)
    strcpy(syn0_base_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-syn1-base", argc, argv)) > 0)
    strcpy(syn1_base_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-syn0", argc, argv)) > 0)
    strcpy(syn0_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-syn1", argc, argv)) > 0)
    strcpy(syn1_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-syn0-base-out", argc, argv)) > 0)
    strcpy(syn0_base_out_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-syn1-base-out", argc, argv)) > 0)
    strcpy(syn1_base_out_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-syn0-out", argc, argv)) > 0)
    strcpy(syn0_out_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-syn1-out", argc, argv)) > 0)
    strcpy(syn1_out_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-syn1-net-out", argc, argv)) > 0)
    strcpy(syn1_net_out_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-lambda-base", argc, argv)) > 0)
    lambda_base = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambda-net", argc, argv)) > 0)
    lambda_net = atof(argv[i + 1]);
  assert(negative > 0);
  vocab =
      (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_net =
      (struct vocab_word *)calloc(vocab_max_size_net, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  vocab_hash_net = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) *
                      MAX_EXP);  // Precompute the exp() table
    expTable[i] =
        expTable[i] / (expTable[i] + 1);  // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
