

// MSVC defines this in winsock2.h!?
typedef struct wintimeval {
    long tv_sec;
    long tv_usec;
} wintimeval;

int gettimeofday(wintimeval* tp);