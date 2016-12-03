
#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"

extern int layer_size;

void load(BPNN *net)
{
  float *units;
  int nr, i, k;
  // nc, imgsize, j, 
  nr = layer_size;
  
  // imgsize = nr * nc;
  units = net->input_units;

  k = 1;
  for (i = 0; i < nr; i++) {
    units[k] = (float) rand()/RAND_MAX ;
    k++;
  }
}
