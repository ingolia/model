#include <stdio.h>
#include <stdlib.h>

#include "grid2d.h"

int main(void)
{
  grid2d *g1 = grid2d_new_rectangle(17, 23);
  grid2d *g2 = grid2d_new_rectangle(23, 17);
  grid2d *g3 = grid2d_new_rectangle(19, 19);

  validate_grid2d(g1);
  validate_grid2d(g2);
  validate_grid2d(g3);

  grid2d_free(g1);
  grid2d_free(g2);
  grid2d_free(g3);
}
