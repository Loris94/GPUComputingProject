#pragma once

#define T int

typedef struct vector {
    T* data;
    int capacity;
    int size;
} vector;

struct vector* vec_init();
//struct vector* vec_init(int size);
void vec_push(struct vector* vec, T item);
T vec_get(struct vector* vec, int i);
void vec_set(struct vector* vec, int i, T item);
void vec_free(struct vector* vec);
