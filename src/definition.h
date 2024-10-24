const int blockSize = 256;
const int nnz = 56825529;
const int n_points = 11437831;
const int n_points_red = 5666455;
const int n_points_black = 5666400;
const int n_points_edge = n_points - (n_points_red + n_points_black);
int loop_count = 100;
double omega = 1.4; // Define relaxation parameter