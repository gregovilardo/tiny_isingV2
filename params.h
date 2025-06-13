#pragma once

#ifndef L
#define L 512 // linear system size
#endif

#ifndef TEMP_INITIAL
#define TEMP_INITIAL 1.5f // initial temperature
#endif

#ifndef TEMP_FINAL
#define TEMP_FINAL 3.0f // final temperature
#endif

#ifndef TEMP_DELTA
#define TEMP_DELTA 0.01f // temperature step
#endif

#ifndef TRAN
#define TRAN 5 // equilibration time
#endif

#ifndef TMAX
#define TMAX 25 // measurement time
#endif

#ifndef DELTA_T
#define DELTA_T 5 // sampling period for energy and magnetization
#endif
