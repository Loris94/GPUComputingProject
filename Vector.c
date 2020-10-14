#include <stdio.h>
#include <stdlib.h>
#include "Vector.h"

#define CAPACITY 5

struct vector* vec_init() {
    vector* vec = (vector*)malloc(sizeof(struct vector));
    vec->capacity = CAPACITY;
    vec->data = (T*)malloc(vec->capacity * sizeof(T));
    vec->size = 0;
    return vec;
}

void vec_push(struct vector* vec, T item) {
    if (vec->size == vec->capacity) {
        
        vec->capacity += CAPACITY;
        vec->data = (T*) realloc(vec->data, vec->capacity * sizeof(T));
    }

    vec->data[vec->size] = item;
    vec->size++;
}

T vec_get(struct vector* vec, int i) {
    if (i >= vec->size) {
        return -1; // not existing!
    }
    else {
        return vec->data[i];
    }
}

void vec_set(struct vector* vec, int i, T item) {
    int n;

    if (i >= vec->capacity) {
        n = (i - vec->capacity) / CAPACITY + 1;
        vec->capacity += n * CAPACITY;
        vec->data = (T*)realloc(vec->data, vec->capacity * sizeof(T));
        vec->size = i + 1;
        vec->data[i] = item;
    }
    else {
        if (i >= vec->size) vec->size = i + 1;
        vec->data[i] = item;
    }
}

void vec_free(struct vector* vec) {
    free(vec->data);
    free(vec);
}

