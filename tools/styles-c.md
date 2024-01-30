# C/C++ Programming
 
## Style Guide

### Variables

- variables should have scope as narrow as possible for how they are used
- communication between functions should be via parameters and return values rather than through global variables

### Identifiers
- Identifiers should be lower case, with underscores between words in multi-word identifiers.
- Single-letter identifiers are unacceptable except for loop counters.
- Constants defined with #define should be all upper case and with underscores to separate words.

### Indentation and other Spacing
- The contents of a block should be indented further than the containing block. You may use whatever nonzero indentation you wish as long as you are consistent.
- Put a space between function arguments and around binary operators (so f(x + y, z), not f(x+y,z)). 
- You may put the opening brace of a block at the end of the line or on its own line as long as you are consistent.
- The closing brace should be on its own line.
- Use at least one blank line between function definitions.
- Use blank lines with functions to group statements that work together to implement a single step of an algorithm.

### Comments
- Each function declaration should be preceded by a comment describing
    - any preconditions (what needs to be true in order for the function to do what the comment claims it will do);
    - the postconditions (what it does, including effects on data pointed to by any pointer arguments and (when used properly) global variables, and the return value;
    - the caller's responsibility for releasing any dynamically allocated memory or other resources allocated by the function; and
    - the purpose of each argument.
- Within functions, comments should be used to indicate which lines of code implement which steps of an algorithm (so will roughly correspond to blank lines within a function) and invariants that are not obvious and are necessary for the subsequent code to be correct.

## Assertions
Every non-trivial C program should `include <assert.h>`, which gives you the assert macro. The assert macro tests if a condition is true and halts your program with an error message if it isn’t.

## Not recommended: debugging output
A tempting but usually bad approach to debugging is to put lots of `printf` statements in your code to show what is going on. The problem with this compared to using `assert` is that there is no built-in test to see if the output is actually what you’d expect. 

If you really need to use `printf` or something like it for debugging output, here are a few rules of thumb to follow to mitigate the worst effects:

- Use `fprintf(stderr, ...)` instead of `printf(...)`; this allows you to redirect your program’s regular output somewhere that keeps it separate from the debugging output.
- Wrap your debugging output in an `#ifdef` so you can turn it on and off easily.

## Valgrind
The valgrind program can be used to detect some (but not all) common errors in C programs that use pointers and dynamic storage allocation. 
```
valgrind ./my-program arg1 arg2 < test-input
```
This will run your program and produce a report of any allocations and de-allocations it did. It will also warn you about common errors like using unitialized memory, dereferencing pointers to strange places, writing off the end of blocks allocated using `malloc`, or failing to free blocks.

You can suppress all of the output except errors using the `-q` option, like this:

```
valgrind -q ./my-program arg1 arg2 < test-input
```
You can also turn on more tests, e.g.
```
valgrind -q --tool=memcheck --leak-check=yes ./my-program arg1 arg2 < test-input
```

### Compilation flags
You can run valgrind on any program (try valgrind ls); it does not require special compilation. However, the output of valgrind will be more informative if you compile your program with debugging information turned on using the `-g` or `-g3` flags (this is also useful if you plan to watch your program running using gdb, ).
