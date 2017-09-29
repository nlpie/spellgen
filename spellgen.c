//  Copyright 2013 Google Inc. All Rights Reserved.
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

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include "strmap.h"

const long long max_size = 2000;         // max length of strings
const long long N = 1000;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

typedef int (*compfn)(const void*, const void*);

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

const char* stoplist[] = {"a","an","and","are","as","at","be","by","for","from","has","he","in","is","it","its","of","on","that","the","to","was","were","will","with","mg","prn"}; 

struct Candidates {
  char  word[100];
  float cosine;
  int cfreq;
  int ed;
  float score;
};

// a test instance to hold information from NLM ANN files
struct TestInstance {
  char id[5];
  char incorrect[200];
  char correct[200];
};

int levenshtein(char *s1, char *s2) {
  //printf("LL: |%s|%s|",s1,s2);
  unsigned int x, y, s1len, s2len;
  s1len = strlen(s1);
  s2len = strlen(s2);
  unsigned int matrix[s2len+1][s1len+1];
  matrix[0][0] = 0;
  for (x = 1; x <= s2len; x++)
    matrix[x][0] = matrix[x-1][0] + 1;
  for (y = 1; y <= s1len; y++)
    matrix[0][y] = matrix[0][y-1] + 1;
  for (x = 1; x <= s2len; x++)
    for (y = 1; y <= s1len; y++)
      matrix[x][y] = MIN3(matrix[x-1][y] + 1, matrix[x][y-1] + 1, matrix[x-1][y-1] + (s1[y-1] == s2[x-1] ? 0 : 1));

  return(matrix[s2len][s1len]);
}

// sort multiple arrays
int sortCandidatesScore(struct Candidates* elem1, struct Candidates* elem2)
{

  /* Cosine */
  if ( elem1->score < elem2->score)
    return 1;
  else if ( elem1->score > elem2->score )
    return -1;
  else
    return 0;
}


int sortCandidatesCosine(struct Candidates* elem1, struct Candidates* elem2)
{
 
  /* Cosine */
  if ( elem1->cosine < elem2->cosine)
    return 1;
  else if ( elem1->cosine > elem2->cosine )
    return -1;
  else 
    return 0;
}

int sortCandidatesFreq(struct Candidates* elem1, struct Candidates* elem2)
{

  /* Candidate Frequency */

  if ( elem1->cfreq < elem2->cfreq )
    return 1;
  else if ( elem1->cfreq > elem2->cfreq )
    return -1;
  else
    return 0;

}

int sortCandidatesED(struct Candidates* elem1, struct Candidates* elem2)
{

  /* Edit Distance*/

  if ( elem1->ed < elem2->ed )
    return -1;
  else if ( elem1->ed > elem2->ed )
    return 1;
  else
    return 0; /* definitely! */

}



// routine to read a field from a space separated file
const char* getfield(char* line, int num, int num1)
{
  const char* tok;
  char ret[100];

  int cnt = 0;

  tok = strtok(line," ");

  while(tok != NULL){

    if(num == cnt){
      strcpy(ret, tok);
      strcat(ret,":");
    }
    if(num1 == cnt){
      strcat(ret, tok);
      return ret;
    }

    tok = strtok(NULL," ");
    cnt++;
  }

}


// iterate over hash
static void iter(const char *key, const char *value, const void *obj)
{
  printf("key: %s value: %s\n", key, value);
}

// simple edit distance
static int ldistance (const char * word1,
                     int len1,
                     const char * word2,
                     int len2)
{
  int matrix[len1 + 1][len2 + 1];
  int i;
  for (i = 0; i <= len1; i++) {
    matrix[i][0] = i;
  }
  for (i = 0; i <= len2; i++) {
    matrix[0][i] = i;
  }
  for (i = 1; i <= len1; i++) {
    int j;
    char c1;

    c1 = word1[i-1];
    for (j = 1; j <= len2; j++) {
      char c2;

      c2 = word2[j-1];
      if (c1 == c2) {
	matrix[i][j] = matrix[i-1][j-1];
      }
      else {
	int delete;
	int insert;
	int substitute;
	int minimum;

	delete = matrix[i-1][j] + 1;
	insert = matrix[i][j-1] + 1;
	substitute = matrix[i-1][j-1] + 1;
	minimum = delete;
	if (insert < minimum) {
	  minimum = insert;
	}
	if (substitute < minimum) {
	  minimum = substitute;
	}
	matrix[i][j] = minimum;
      }
    }
  }
  return matrix[len1][len2];
}



int main(int argc, char **argv) {
  FILE *f, *ff;
  char st1[max_size];
  char *bestw[N];
  char file_name[max_size], st[100][max_size], freq_file[max_size];
  float dist, len, bestd[N], vec[max_size];
  long long words, size, a, b, c, d, cn, u, bi[100];
  char ch;
  float *M;
  char *vocab;

  // levenstein's distance
  long long ldist=0;

  // string map
  StrMap *sm;
  int result;
  int topN = 20;


  if (argc < 3) {
    printf("Usage: ./spellgen <FILE> <FREQ> <topN> \nwhere FILE contains word projections in the BINARY FORMAT. FREQ contains word frequencies in plain text. topN - number of top scoring items to return\n");
    return 0;
  }

  strcpy(file_name, argv[1]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }

  strcpy(freq_file, argv[2]);
  ff = fopen(freq_file, "rb");
  if (ff == NULL) {
    printf("Frequency file not found\n");
    return -1;
  }

  topN = atoi(argv[3]);

  char line[1024];
  // allocate a map of N elements
  sm = sm_new(10000000);
  if (sm == NULL) {
    /* Handle allocation failure... */
    printf('Could not allocate a hash map\n');
    return 0;
  }

  while(fgets(line, 1024, ff))
    {
     
      char* tmp = strdup(line);
      char* pair = getfield(tmp, 0, 1);
      char* word = strtok(pair,":");
      char* count = strtok(NULL,":");
      sm_put(sm, word, count);
      //printf("Word: %s  %s\n", word, count);
    }

  //sm_enum(sm, iter, NULL);

  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));

  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }

  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }

  fclose(f);

  while (1) {
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    printf("Enter word or sentence (EXIT to break): ");
    a = 0;
    while (1) {
      st1[a] = fgetc(stdin);
      if ((st1[a] == '\n') || (a >= max_size - 1)) {
        st1[a] = 0;
        break;
      }
      a++;
    }
    if (!strcmp(st1, "EXIT")) break;
    cn = 0;
    b = 0;
    c = 0;
    while (1) {
      st[cn][b] = st1[c];
      b++;
      c++;
      st[cn][b] = 0;
      if (st1[c] == 0) break;
      if (st1[c] == ' ') {
        cn++;
        b = 0;
        c++;
      }
    }
    cn++;
    for (a = 0; a < cn; a++) {
      for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st[a])) break;
      if (b == words) b = -1;
      bi[a] = b;
      printf("\nWord: %s  Position in vocabulary: %lld\n", st[a], bi[a]);
      if (b == -1) {
        printf("Out of dictionary word!\n");
        break;
      }
    }
    if (b == -1) continue;

    printf("\n                                              Word                 Cosine      ED    Frequency         Score\n------------------------------------------------------------------------\n");
    for (a = 0; a < size; a++) vec[a] = 0;
    for (b = 0; b < cn; b++) {
      if (bi[b] == -1) continue;
      for (a = 0; a < size; a++) vec[a] += M[a + bi[b] * size];
    }
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;
    for (a = 0; a < N; a++) bestd[a] = -1;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    for (c = 0; c < words; c++) {
      a = 0;
      for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
      if (a == 1) continue;
      dist = 0;
      for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];
      for (a = 0; a < N; a++) {
        if (dist > bestd[a]) {
          for (d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw[d], bestw[d - 1]);
          }
          bestd[a] = dist;
          strcpy(bestw[a], &vocab[c * max_w]);
          break;
        }
      }
    }
   
    
    //struct Candidates *listCosine;
    //struct Candidates *listFreq;
    //struct Candidates *listED;
    struct Candidates *listScore;
    //listCosine = (struct Candidates *)malloc((N * sizeof(struct Candidates)));
    //listFreq = (struct Candidates *)malloc((N * sizeof(struct Candidates)));
    //listED = (struct Candidates *)malloc((N * sizeof(struct Candidates)));
    listScore = (struct Candidates *)malloc((N * sizeof(struct Candidates))); 



    for (a = 0; a < N; a++) 
      {
	const char * wrd2;
	wrd2 = "kidney";
	ldist = 0;
	//ldist = ldistance(bestw[a],strlen(bestw[a]),st[a],strlen(st[a]));

	ldist = levenshtein(bestw[a], st[0]);

	// get the frequency
	char buf[255];
	int cfreq = 0;
	result = sm_get(sm, bestw[a], buf, sizeof(buf));
        if (result == 0) {
	  /* Handle value not found... */
	}else{
	  cfreq = atoi(buf);
	}

	char buf1[255];
	int tfreq = 0;
	result = sm_get(sm, st[0], buf1, sizeof(buf1));
        if (result == 0) {
          /* Handle value not found... */
        }else{
	  tfreq = atoi(buf1);
	}


	float score = 0.0;

	// formula to find the correctly spelled word
	//score = log10f((pow(10,bestd[a])*(float)cfreq)/((float)ldist*(float)ldist));
	
	// formular to generate misspellings from a correctly spelled word
	score = log10f(pow(10,bestd[a])/((float)ldist*(float)ldist));
	
	//printf("%50s\t\t%f\t%lld\t%d\t%d\t%f\n", bestw[a], bestd[a], ldist, tfreq, cfreq, score);
	//struct Candidates cand;

	//scanf("%s %d %d", list[i].month, &list[i].day, &list[i].year);
	
	/*strcpy(listCosine[a].word,bestw[a]);
	listCosine[a].cosine = bestd[a];
	listCosine[a].ed = ldist;
	listCosine[a].cfreq = cfreq;

	strcpy(listFreq[a].word,bestw[a]);
        listFreq[a].cosine = bestd[a];
        listFreq[a].ed = ldist;
        listFreq[a].cfreq = cfreq;

	strcpy(listED[a].word,bestw[a]);
        listED[a].cosine = bestd[a];
        listED[a].ed = ldist;
        listED[a].cfreq = cfreq;
	*/

	strcpy(listScore[a].word,bestw[a]);                                                                                                                                                                                                               
        listScore[a].cosine = bestd[a];                                                                                                                                                                                                 
        listScore[a].ed = ldist;                                                                                                                                                                                  
        listScore[a].cfreq = cfreq;     
	listScore[a].score = score;

      }

    qsort(listScore, N, sizeof(struct Candidates), (compfn)sortCandidatesScore);

    int isastop = 0;

    for(a = 0; a < topN; a++)                                                                                                                                                                                                      
      {	  

	for(u = 0; u < sizeof(stoplist)/sizeof(stoplist[0]); u++){
	  if(strcmp(stoplist[u],listScore[a].word) == 0){
	    isastop = 1;
	  }
	}

	if(isastop == 0){
	  printf("%50s\t\t%f\t%d\t%d\t%f\n", listScore[a].word,listScore[a].cosine,listScore[a].ed,listScore[a].cfreq, listScore[a].score);
	}else{
	  //printf("%50s\tis a stopword",listScore[a].word);
	}
	isastop = 0;
      }           


    /*qsort(listCosine, N, sizeof(struct Candidates), (compfn)sortCandidatesCosine);
    qsort(listFreq, N, sizeof(struct Candidates), (compfn)sortCandidatesFreq);
    qsort(listED, N, sizeof(struct Candidates), (compfn)sortCandidatesED);

    for(a = 0; a < N; a++)
      {

	printf("%s %f %d %d  %s %f %d %d %s %f %d %d\n", listCosine[a].word,listCosine[a].cosine,listCosine[a].ed,listCosine[a].cfreq, listFreq[a].word,listFreq[a].cosine,listFreq[a].ed,listFreq[a].cfreq, listED[a].word,listED[a].cosine,listED[a].ed,listED[a].cfreq);
	if ( strcmp(listED[a].word,listFreq[a].word) == 0){
	  //printf("\n\n%s\t\t\t\t%f %d %d\n", listED[a].word, listED[a].cosine, listED[a].ed, listED[a].cfreq);
	  //break;
	}
      }
    */

  }
  return 0;
}
