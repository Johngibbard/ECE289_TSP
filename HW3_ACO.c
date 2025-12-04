#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// Parameters (tweak as needed)
#define DEFAULT_NUM_NODES 1000
#define DEFAULT_NUM_ANTS 10
#define DEFAULT_RUN_SECONDS 60
#define DEFAULT_NUM_RUNS 10

// ---------------------------------------------------------------------------
// Configuration (edit these values instead of passing command-line args)
// ---------------------------------------------------------------------------
// Path to graph file (edges as: nodeA nodeB distance) - edit to point to your file
static const char *GRAPH_FILE = "Graphs\\TSP_1000_euclidianDistance.txt";
// number of nodes in the graph
static const int CONFIG_NUM_ANTS = DEFAULT_NUM_ANTS;

// ACO constants
const double RHO = 0.06;           // evaporation rate
const double PHEROMONE_INIT = 1.0; // initial pheromone
const double PHEROMONE_MIN = 1e-18;
const double PHEROMONE_MAX = 0.1;
const double PHEROMONE_Q = 1;      // total pheromone budget per iteration
const double GAMMA_EXP = 1e7;      // exponential favoritism
const double ALPHA = 2;            // pheromone importance
const double BETA = 2;             // heuristic importance (1/distance)

// (stagnation detection / diversification removed)

// helpers for 2D arrays
static inline int idx(int n, int i, int j) { return i * n + j; }

// read edge list (i j dist) into distance matrix, nodes are 1-indexed in file
int read_graph(const char *path, double *dist, int n)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        int a,b;
        double d;
        if (sscanf(line, "%d %d %lf", &a, &b, &d) == 3) {
            if (a >= 1 && a <= n && b >=1 && b <= n) {
                dist[idx(n,a-1,b-1)] = d;
                dist[idx(n,b-1,a-1)] = d;
            }
        }
    }
    fclose(f);
    return 0;
}

// compute distance with fallback symmetric check
static inline double get_dist(int n, double *dist, int i, int j)
{
    /* read_graph stores both directions; return directly */
    return dist[idx(n,i,j)];
}

// compute tour length; tour is length n+1 with last == first
static inline double tour_length(int n, double *dist, int *tour, int tour_len)
{
    double s = 0.0;
    for (int i = 0; i < tour_len - 1; ++i) {
        s += get_dist(n, dist, tour[i], tour[i+1]);
    }
    return s;
}

// 2-opt improvement (in-place). tour is size n+1 with last=first
void two_opt(int n, double *dist, int *tour, int tour_len)
{
    if (tour_len < 4) return;
    int improved = 1;
    while (improved) {
        improved = 0;
        for (int i = 0; i < tour_len - 3; ++i) {
            for (int j = i + 2; j < tour_len - 1; ++j) {
                // avoid replacing the closing edge
                if (i == 0 && j == tour_len - 2) continue;
                int a = tour[i];
                int b = tour[i+1];
                int c = tour[j];
                int d = tour[j+1];
                double cur = get_dist(n, dist, a, b) + get_dist(n, dist, c, d);
                double nw = get_dist(n, dist, a, c) + get_dist(n, dist, b, d);
                if (nw + 1e-12 < cur) {
                    // reverse segment i+1..j
                    int l = i+1, r = j;
                    while (l < r) {
                        int tmp = tour[l]; tour[l] = tour[r]; tour[r] = tmp;
                        l++; r--;
                    }
                    improved = 1;
                }
            }
        }
    }
}

// simple thread-local linear psuedo-random number generator
static inline unsigned int lcg_next(unsigned int *state)
{
    *state = (*state) * 1103515245u + 12345u;
    return (*state >> 16) & 0x7fffu;
}
static inline double lcg_double(unsigned int *state)
{
    return (double)lcg_next(state) / (double)0x7fffu;
}

// Compute MST graph to get a lower bound using Prim's algorithm
double prim_mst(int n, double *dist) {
    double *minEdge = malloc(n * sizeof(double));
    int *used = calloc(n, sizeof(int));
    if (!minEdge || !used) { free(minEdge); free(used); return 1e300; }
    for (int i = 0; i < n; ++i) { minEdge[i] = 1e300; used[i] = 0; }
    minEdge[0] = 0.0;
    double total = 0.0;
    for (int iter = 0; iter < n; ++iter) {
        int v = -1;
        for (int j = 0; j < n; ++j) if (!used[j] && (v == -1 || minEdge[j] < minEdge[v])) v = j;
        if (v == -1) { total = 1e300; break; }
        used[v] = 1;
        total += minEdge[v];
        for (int to = 0; to < n; ++to) {
            if (used[to]) continue;
            double d = dist[idx(n, v, to)];
            if (d > 0.0 && d < minEdge[to]) minEdge[to] = d;
        }
    }
    free(minEdge); free(used);
    return total;
}

// print function with flush
static void print_status(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    fflush(stdout);
}

// create a random ant tour using pheromone^alpha * heuristic^beta
// rng is a pointer to a thread-local RNG state (unsigned int)
// returns tour array (length n+1) and length via out parameter
void construct_ant(int n, double *dist, double *pher, int *tour_out, double *out_len, unsigned int *rng)
{
    int *unvisited = (int*)malloc(n * sizeof(int)); // create unvisited set 
    int uncount = n;
    for (int i = 0; i < n; ++i) unvisited[i] = 1;
    int cur = (int)(lcg_next(rng) % (unsigned int)n); // pseudo-random start
    tour_out[0] = cur;
    unvisited[cur] = 0; uncount--;
    int pos = 1;
    while (uncount > 0) {
        // compute probabilities over unvisited
        double denom = 0.0;
        double *probs = (double*)malloc(n * sizeof(double));
        for (int j = 0; j < n; ++j) probs[j] = 0.0;
        for (int j = 0; j < n; ++j) {
            if (!unvisited[j]) continue;
            double d = get_dist(n, dist, cur, j);
            if (d <= 0) continue;
            double tau = pher[idx(n,cur,j)];
            double heur = 1.0 / d;
            double p = pow(tau, ALPHA) * pow(heur, BETA);
            probs[j] = p;
            denom += p;
        }
        if (denom <= 0.0) break; // cannot continue
        double r = lcg_double(rng) * denom;
        double acc = 0.0;
        int next = -1;
        for (int j = 0; j < n; ++j) {
            if (probs[j] <= 0.0) continue;
            acc += probs[j];
            if (r <= acc) { next = j; break; }
        }
        free(probs);
        if (next == -1) break;
        tour_out[pos++] = next;
        unvisited[next] = 0; uncount--; cur = next;
    }
    // append any remaining unvisited, then close the tour
    for (int j = 0; j < n; ++j) if (unvisited[j]) { tour_out[pos++] = j; unvisited[j]=0; }
    tour_out[pos++] = tour_out[0];
    *out_len = tour_length(n, dist, tour_out, pos);
    free(unvisited);
}

int main(void)
{
    const char *graph_path = GRAPH_FILE;
    int N = DEFAULT_NUM_NODES;
    int NUM_ANTS = DEFAULT_NUM_ANTS;
    int run_seconds = DEFAULT_RUN_SECONDS;
    if (N <= 2) { fprintf(stderr, "num_nodes must be >= 3\n"); return 1; }
    if (NUM_ANTS <= 0) NUM_ANTS = DEFAULT_NUM_ANTS;

    // allocate distance matrix and read graph first
    double *dist = (double*)calloc(N * N, sizeof(double));
    if (read_graph(graph_path, dist, N) != 0) {
        fprintf(stderr, "Failed to read graph file: %s\n", graph_path);
        return 1;
    }

    // compute an MST lower bound for the TSP (cheap and useful)
    double mst_lb = prim_mst(N, dist);
    if (mst_lb >= 1e299) print_status("Warning: MST not found (graph may be disconnected)\n");

    // Optional: number of independent ACO runs (parallel). Default taken from config
    int NUM_RUNS = DEFAULT_NUM_RUNS;
    if (DEFAULT_NUM_RUNS < 1) NUM_RUNS = 1;


    int *best_overall_tour = (int*)malloc((N+1) * sizeof(int));
    double best_overall_len = 1e300;

    // start running trials 
    print_status("Starting run (time limit %d seconds)\n", run_seconds);
        // Save the best overall path at the start of this trial (checkpoint)
        double initial_best_len = best_overall_len;
            if (initial_best_len < 1e299) {
            // Writing tours to files has been disabled.
            print_status("Skipping save of start-of-run best (file output disabled) (length=%.6f)\n", initial_best_len);
        } else {
            print_status("No previous overall best to save\n");
            }
    int *global_best_tour = (int*)malloc((N+1) * sizeof(int));
    double global_best_len = 1e300;
    double *per_run_best = (double*)malloc(NUM_RUNS * sizeof(double));

    for (int i = 0; i < NUM_RUNS; ++i) per_run_best[i] = 1e300;
#pragma omp parallel num_threads(NUM_RUNS)
        {
            int tid = omp_get_thread_num();
            if (tid >= NUM_RUNS) {
                /* extra threads do nothing */
            } else {

            // local pheromone, tours and RNG
            double *pher_local = (double*)calloc(N * N, sizeof(double));
            for (int i = 0; i < N*N; ++i) pher_local[i] = (dist[i] > 0.0) ? PHEROMONE_INIT : 0.0;

            int *best_tour = (int*)malloc((N+1) * sizeof(int));
            double best_len = 1e300;

            double *ant_lengths = (double*)malloc(NUM_ANTS * sizeof(double));
            int **ant_tours = (int**)malloc(NUM_ANTS * sizeof(int*));
            for (int a = 0; a < NUM_ANTS; ++a) ant_tours[a] = (int*)malloc((N+1) * sizeof(int));

            unsigned int rng = (unsigned int)time(NULL) ^ (unsigned int)(tid * 2654435761u);
            time_t t0 = time(NULL);
            double last_print = -1.0;
            int iter = 0;
            while (difftime(time(NULL), t0) < (double)run_seconds) {
                for (int a = 0; a < NUM_ANTS; ++a) construct_ant(N, dist, pher_local, ant_tours[a], &ant_lengths[a], &rng);

                int best_idx = 0; for (int a = 1; a < NUM_ANTS; ++a) if (ant_lengths[a] < ant_lengths[best_idx]) best_idx = a;
                // if (iter < 1000 || iter % 100 == 0) 
                two_opt(N, dist, ant_tours[best_idx], N+1);
                ant_lengths[best_idx] = tour_length(N, dist, ant_tours[best_idx], N+1);

                double L_best = ant_lengths[best_idx];
                double *raw = (double*)malloc(NUM_ANTS * sizeof(double));
                double sum_raw = 0.0;
                for (int a = 0; a < NUM_ANTS; ++a) {
                    double L = ant_lengths[a];
                    double rel = (L - L_best) / (L_best > 0 ? L_best : 1.0);
                    double arg = -GAMMA_EXP * rel; if (arg < -700.0) arg = -700.0; if (arg > 700.0) arg = 700.0;
                    raw[a] = exp(arg); sum_raw += raw[a];
                }
                if (sum_raw <= 0.0) sum_raw = 1.0;

                for (int i = 0; i < N*N; ++i) { 
                    pher_local[i] *= (1.0 - RHO);
                    if (pher_local[i] < PHEROMONE_MIN) pher_local[i] = PHEROMONE_MIN; 

                }

                for (int a = 0; a < NUM_ANTS; ++a) {
                    double w = raw[a] / sum_raw;
                    double delta = PHEROMONE_Q * w / (ant_lengths[a] > 0 ? ant_lengths[a] : 1.0);
                    int *t = ant_tours[a];
                    for (int i = 0; i < N; ++i) {
                        int u = t[i], v = t[i+1]; int id = idx(N,u,v), id2 = idx(N,v,u);
                        if (dist[id] > 0.0) { pher_local[id] += delta; if (pher_local[id] > PHEROMONE_MAX) pher_local[id] = PHEROMONE_MAX; }
                        if (dist[id2] > 0.0) { pher_local[id2] += delta; if (pher_local[id2] > PHEROMONE_MAX) pher_local[id2] = PHEROMONE_MAX; }
                    }
                }
                free(raw);
                if (ant_lengths[best_idx] < best_len) {
                    best_len = ant_lengths[best_idx];
                    memcpy(best_tour, ant_tours[best_idx], (N+1)*sizeof(int));
                }

                double sum = 0.0; for (int a = 0; a < NUM_ANTS; ++a) sum += ant_lengths[a]; double avg = sum / NUM_ANTS;
                double elapsed = difftime(time(NULL), t0);
                if (elapsed - last_print >= 1.0) { // print at most once per second
                    print_status("[Run %d] Time: %.6f seconds Iter %d: best=%.6f avg=%.6f local_best=%.6f\n", tid, elapsed, iter, ant_lengths[best_idx], avg, best_len);
                    last_print = elapsed;
                }
                iter++;
            }
            #pragma omp critical
            {
                if (best_len < global_best_len) { global_best_len = best_len; memcpy(global_best_tour, best_tour, (N+1)*sizeof(int)); }
            }
                if (tid < NUM_RUNS) per_run_best[tid] = best_len;
                #pragma omp barrier
                for (int a = 0; a < NUM_ANTS; ++a) free(ant_tours[a]); free(ant_tours); free(ant_lengths); free(best_tour); free(pher_local);
            }
        // end of parallel region
        }

        /* compute average of per-run bests (exclude unused entries) and
         * print the final best once after all threads have finished */
        double sum_runs = 0.0; int used_runs = 0;
        for (int i = 0; i < NUM_RUNS; ++i) {
            if (per_run_best[i] < 1e299) { sum_runs += per_run_best[i]; used_runs++; }
        }
        double avg_runs = (used_runs > 0) ? sum_runs / used_runs : 0.0;

        print_status("Run finished: best=%.6f\n", global_best_len);
        // print the best path for this run (1-indexed)
        print_status("Best path (run): ");
        for (int i = 0; i < N+1; ++i) {
            if (i == 0) print_status("%d", global_best_tour[i] + 1);
            else print_status(" %d", global_best_tour[i] + 1);
        }
        print_status("\n");
    // update overall best tour
    if (global_best_len < best_overall_len) { best_overall_len = global_best_len; memcpy(best_overall_tour, global_best_tour, (N+1)*sizeof(int)); }
    free(global_best_tour);
    free(per_run_best);
    
    // end trials loop
    // Print overall summary for single run
    print_status("Completed run. Best overall length: %.6f\n", best_overall_len);
    if (mst_lb < 1e299 && best_overall_len < 1e299) {
        double gap = 100.0 * (best_overall_len - mst_lb) / mst_lb;
        print_status("MST LB: %.6f  gap vs best overall: %.3f%%\n", mst_lb, gap);
    } else if (mst_lb >= 1e299) {
        print_status("MST LB unavailable (graph may be disconnected)\n");
    }

    // print the best tour (path) to stdout exactly once.
    if (best_overall_len < 1e299) {
        print_status("Best overall length: %.6f (file output disabled)\n", best_overall_len);
        // path already printed above at run finish; do not duplicate
    }
    free(best_overall_tour);
    free(dist);
    return 0;
    } 
    

