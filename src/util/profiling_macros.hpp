#pragma once
 
#ifdef BUBBLEFINDER_INSTRUMENT
    #define BF_INSTR(...) __VA_ARGS__
#else
    #define BF_INSTR(...) 
#endif
 