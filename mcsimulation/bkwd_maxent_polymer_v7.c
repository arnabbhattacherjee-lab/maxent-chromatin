/* remc_polymer_targetT_trace_v7_forward_fixed_lambda.c
 *  *
 *   * Forward MaxEnt polymer simulation with FIXED input lambda
 *    * MPI + OpenMP  ·  incremental MC with BVH-accelerated excluded volume
 *     *
 *      * Key properties
 *       * --------------
 *        * 1. lambda is read from LAMDA_FORWARD and NEVER updated.
 *         * 2. Annealed debias/rebias schedule changes only bias_beta and T.
 *          * 3. old_energy is rebased cheaply from stored components when bias_beta changes.
 *           * 4. Contact map is computed only:
 *            *      (a) at the end of each rebias block
 *             *      (b) once at the very end
 *              *
 *               * Compile:
 *                *   mpicc -O3 -march=native -fopenmp -D_GNU_SOURCE \
 *                 *         bkwd_maxent_polymer_v7.c -lm -o bkwd_maxent
 *                  *
 *                   * Run:
 *                    *   mpirun -np 16 ./bkwd_maxent
 *                     */

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <mpi.h>
#include "sys.h"

#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_max_threads(void) { return 1; }
static inline int omp_get_thread_num(void)  { return 0; }
static inline void omp_set_dynamic(int x)   { (void)x; }
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Simulation constants ──────────────────────────────────────────── */
#define MAX_MOVES                 139800
#define NUM_BEADS 9678        // Number of beads in the polymer
#define NUM_ANGLE 7183        // Number of angles in the polymer
#define NUM_SEGMENTS              1001

/* Bias / debiasing schedule */
#define RB_STEP                  1200
#define DB_STEP                   200
#define StepslowBBupdate          100

/* LJ / BVH */
#ifndef MAX_RADIUS
#define MAX_RADIUS 5.0
#endif
#define MAX_SIGMA  (2.0 * MAX_RADIUS)
#define MAX_RCUT   (2.5 * MAX_SIGMA)

#define MAX_BEADS_IN_LEAF          8
#define BVH_POOL_SIZE           8192
#define REBASE_INTERVAL          200

/* MaxEnt iteration control */
double **prevSimC = NULL;      /* previous block-averaged simulated contact map */
double etadiff    = 0.01;      /* lambda learning rate; tune if needed */

/* how often to update lambda */
#define LAMBDA_UPDATE_INTERVAL 20

/* Persistent per-walker BVH */
#define WBVH_POOL_SIZE         (4 * (NUM_BEADS / MAX_BEADS_IN_LEAF + 1))

/* Per-thread candidate buffer for BVH-queried incremental LJ */
#define MAX_CANDS         NUM_BEADS

/* Max beads moved in one crankshaft */
#define MAX_CRANK_BEADS          24

/* Safe upper bound for affected segments in local crankshaft update.
 *  * If exceeded, fall back to full COM/raw-bias recomputation. */
#define MAX_AFFECTED_SEGS        16

/* ══════════════════════ MPI globals ══════════════════════════════════ */
int mpi_rank = 0, mpi_size = 1;
int ms_start = 0, ms_end   = 0;
int local_Tsnap = 0;

#define LIDX(ms) ((ms) - ms_start)

/* ══════════════════════ Physics globals ══════════════════════════════ */
double ref_dist2   = 625.0;
double ref_dist    = 25.0;
double kharmonic   = 0.1;
double kbend       = 20.0;
double epsilon     = 0.05;
double bias_beta   = 0.0005;
double bb_MAX      = 5.0;
double TEMPERATURE = 1.0;

int bias_phase_state = 0;
int phase_start_step = 0;

/* Tsnap is runtime-global but fixed here */
int Tsnap = 6000;

/* Move probabilities
 *  * ran < probA          => pivot
 *   * probA <= ran < probB => gaussian
 *    * ran >= probB         => crankshaft
 *     *
 *      * Current setting: ~33/34/33
 *       * For 10/20/70 use: probA=0.10, probB=0.30
 *        */
double probA = 0.33;
double probB = 0.67;

/* ══════════════════════ Main simulation arrays ══════════════════════ */
double  *old_energy      = NULL;
double  *new_energy      = NULL;
double ***old_positions  = NULL;
double ***positions      = NULL;
double ***segment_com    = NULL;

double **lamda           = NULL;
double **expC            = NULL;
double **simC            = NULL;

double *rad              = NULL;
double *bref             = NULL;
int    *P1               = NULL;
int    *P2               = NULL;
int    *P3               = NULL;
int    *btyp             = NULL;

int    *segment_start    = NULL;
int    *segment_end      = NULL;
int    *segment_sizes    = NULL;

/* stored energy components */
double *lj_energy_walker   = NULL;
double *bond_energy_walker = NULL;
double *bend_energy_walker = NULL;
double *raw_bias_walker    = NULL;

/* bead → segment lookup */
int *seg_for_bead = NULL;

/* inverse angle index */
int **bead_to_angles   = NULL;
int  *bead_angle_count = NULL;

/* contact-map buffers */
static int     g_max_threads_cm   = 0;
static double *g_thread_maps_flat = NULL;
static double *g_allreduce_local  = NULL;
static double *g_allreduce_global = NULL;

/* ══════════════════════ BVH structures ═══════════════════════════════ */
typedef struct BVHNode {
    double bbox_min[3], bbox_max[3];
    int start, count;
    struct BVHNode *left, *right;
    int is_leaf;
} BVHNode;

typedef struct {
    BVHNode pool[BVH_POOL_SIZE];
    int used;
} BVHArena;

typedef struct {
    BVHNode pool[WBVH_POOL_SIZE];
    int used;
} WBVHArena;

typedef struct {
    double **pos;
    int axis;
} SortCtx;

static BVHArena  *bvh_arenas       = NULL; /* [max_threads] */
static int      **bvh_idx_bufs     = NULL; /* [max_threads][NUM_BEADS] */

static WBVHArena *walker_wbvh      = NULL; /* [local walkers] */
static int      **walker_wbvh_idx  = NULL; /* [local walkers][NUM_BEADS] */
static BVHNode  **walker_wbvh_roots= NULL; /* [local walkers] */

static int      **thread_cand_buf   = NULL; /* [max_threads][NUM_BEADS] */
static char     **thread_cand_flags = NULL; /* [max_threads][NUM_BEADS] */

/* ══════════════════════ Forward declarations ════════════════════════ */
static inline double contact_weight(double dist);
static double rand_gaussian(unsigned int *seed);
static inline double angle_energy_triplet(double *a, double *b, double *c);
static inline int sphere_box_overlap(double cx, double cy, double cz,
                                     double r,
                                     const double *bmin, const double *bmax);





static inline int pair_is_excluded_topologically(int i1, int i2);
static inline double pair_sigma(int i1,int i2);
static inline double pair_rcut_from_sigma(double sigma);
static inline double pair_rcut(int i1,int i2);
static inline int pair_can_interact(int i1,int i2,double r2);
static inline double soft_repulsion_pair(int i1,int i2,double r2);

static void compute_bounding_box(int *idx,int start,int count,
                                 double *bmin,double *bmax,double **pos);
static int compare_indices_r(const void *a,const void *b,void *ctx);

static BVHNode *build_bvh_arena(int *idx,int start,int count,
                                double **pos,BVHArena *arena,SortCtx *ctx);
static BVHNode *build_wbvh_arena(int *idx,int start,int count,
                                 double **pos,WBVHArena *arena,SortCtx *ctx);

static void bvh_query_sphere(BVHNode *node, int *idx, double **pos,
                             double cx, double cy, double cz, double qr,
                             int excl_lo, int excl_hi,
                             int *cands, int *ncands, char *flags);

static void rebuild_walker_bvh(int ms);

static inline int boxes_overlap(const double *mn1,const double *mx1,
                                const double *mn2,const double *mx2,double m);
static void process_leaf_self(BVHNode *leaf,double *e,double **pos,int *idx);
static void process_leaf_cross(BVHNode *a,BVHNode *b,double *e,double **pos,int *idx);
static void traverse_bvh_cross(BVHNode *a,BVHNode *b,double *e,double **pos,int *idx);
static void traverse_bvh_self(BVHNode *node,double *e,double **pos,int *idx);
static double compute_lj_energy_bvh(double **pos,int num_beads,int tid);

static void update_all_segment_coms(int ms);
static void update_partial_segment_coms(int ms, int seg_lo, int seg_hi);
static double compute_raw_bias_from_coms(int ms);

static double calculate_bond_energy(double ***pos, int lms);
static double calculate_bend_energy(double ***pos, int lms);

static void rebase_total_energy_from_components(int ms_s, int ms_e);
static void recompute_all_energies(int ms, int tid);

static void calculate_contact_map_mpi(int ms_s, int ms_e, int update_lambda);
static void init(void);
static void cleanup(void);

/* MC moves */
static void crankshaft_move(int tid, int snap_id, double ***pos,
                            unsigned int *seed_array,
                            int *out_b1, int *out_b2,
                            int *out_s,  int *out_e);
static void gaussian_displacement(int tid, int snap_id, double ***pos,
                                  unsigned int *seed_array);
static void pivot_move(int tid, int snap_id, double ***pos,
                       unsigned int *seed_array);

/* ══════════════════════ Utility ═════════════════════════════════════ */

static double rand_gaussian(unsigned int *seed)
{
    double u = (double)rand_r(seed) / (double)RAND_MAX;
    double v = (double)rand_r(seed) / (double)RAND_MAX;
    if (u < 1e-12) u = 1e-12;
    return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
}

static inline double angle_energy_triplet(double *a, double *b, double *c)
{
    double v1[3] = { b[0]-a[0], b[1]-a[1], b[2]-a[2] };
    double v2[3] = { c[0]-b[0], c[1]-b[1], c[2]-b[2] };
    double dot   = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
    double m1    = sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
    double m2    = sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2]);

    if (m1 < 1e-8 || m2 < 1e-8) return 0.0;

    double ct = dot / (m1 * m2);
    if (ct >  1.0) ct =  1.0;
    if (ct < -1.0) ct = -1.0;
    return kbend * (1.0 - ct);
}

static inline int sphere_box_overlap(double cx, double cy, double cz,
                                     double r,
                                     const double *bmin, const double *bmax)
{
    double dx = (cx < bmin[0]) ? (bmin[0]-cx) : (cx > bmax[0]) ? (cx-bmax[0]) : 0.0;
    double dy = (cy < bmin[1]) ? (bmin[1]-cy) : (cy > bmax[1]) ? (cy-bmax[1]) : 0.0;
    double dz = (cz < bmin[2]) ? (bmin[2]-cz) : (cz > bmax[2]) ? (cz-bmax[2]) : 0.0;
    return (dx*dx + dy*dy + dz*dz) <= r*r;
}
/* ══════════════════════ Inline guard ═════════════════════════════════════ */

static inline int is_finite3(const double *p)
{
    return isfinite(p[0]) && isfinite(p[1]) && isfinite(p[2]);
}

static void check_positions_finite(int ms, const char *where, int move)
{
    int lms = LIDX(ms);

    for (int b = 0; b < NUM_BEADS; b++) {
        if (!is_finite3(positions[lms][b])) {
            fprintf(stderr,
                    "Rank %d: non-finite coordinate at move %d walker %d bead %d in %s\n",
                    mpi_rank, move, ms, b, where);
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

static void check_energy_finite(double val, const char *name, int move, int ms)
{
    if (!isfinite(val)) {
        fprintf(stderr,
                "Rank %d: non-finite %s at move %d walker %d: %.17g\n",
                mpi_rank, name, move, ms, val);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static inline double contact_weight(double dist)
{
    double k = 5.0;

    if (!isfinite(dist)) return 0.0;

    double x = (dist - ref_dist) / k;

    if (x > 50.0) return 0.0;
    if (x < -50.0) return 1.0;

    return 1.0 / (1.0 + exp(x));
}

/* ══════════════════════ MC moves ════════════════════════════════════ */
static void crankshaft_move(int tid, int snap_id, double ***pos,
                            unsigned int *seed_array,
                            int *out_b1, int *out_b2,
                            int *out_s,  int *out_e)
{
    int max_length = 20;
    int bead1  = rand_r(&seed_array[tid]) % NUM_BEADS;
    int length = 2 + rand_r(&seed_array[tid]) % (max_length - 1);
    int bead2  = bead1 + length;

    if (bead2 >= NUM_BEADS) bead2 = NUM_BEADS - 1;
    if (bead2 - bead1 < 2) {
        *out_b1 = -1; *out_b2 = -1;
        *out_s  = -1; *out_e  = -1;
        return;
    }

    *out_b1 = bead1;
    *out_b2 = bead2;
    *out_s  = bead1 + 1;
    *out_e  = bead2 - 1;

    {
        double dx = pos[snap_id][bead2][0] - pos[snap_id][bead1][0];
        double dy = pos[snap_id][bead2][1] - pos[snap_id][bead1][1];
        double dz = pos[snap_id][bead2][2] - pos[snap_id][bead1][2];
        double r  = sqrt(dx*dx + dy*dy + dz*dz);

        if (r < 1e-8) {
            *out_b1 = -1; *out_b2 = -1;
            *out_s  = -1; *out_e  = -1;
            return;
        }

        double nx = dx/r, ny = dy/r, nz = dz/r;
        double angle = 2.0*M_PI*((double)rand_r(&seed_array[tid])/(double)RAND_MAX);
        double ca = cos(angle), sa = sin(angle), omc = 1.0 - ca;

        double R[3][3];
        R[0][0]=ca+nx*nx*omc;     R[0][1]=nx*ny*omc-nz*sa; R[0][2]=nx*nz*omc+ny*sa;
        R[1][0]=ny*nx*omc+nz*sa;  R[1][1]=ca+ny*ny*omc;    R[1][2]=ny*nz*omc-nx*sa;
        R[2][0]=nz*nx*omc-ny*sa;  R[2][1]=nz*ny*omc+nx*sa; R[2][2]=ca+nz*nz*omc;

        double ox = pos[snap_id][bead1][0];
        double oy = pos[snap_id][bead1][1];
        double oz = pos[snap_id][bead1][2];

        for (int i = bead1 + 1; i < bead2; i++) {
            double x = pos[snap_id][i][0] - ox;
            double y = pos[snap_id][i][1] - oy;
            double z = pos[snap_id][i][2] - oz;

            pos[snap_id][i][0] = R[0][0]*x + R[0][1]*y + R[0][2]*z + ox;
            pos[snap_id][i][1] = R[1][0]*x + R[1][1]*y + R[1][2]*z + oy;
            pos[snap_id][i][2] = R[2][0]*x + R[2][1]*y + R[2][2]*z + oz;
        }
    }
}

static void gaussian_displacement(int tid, int snap_id, double ***pos,
                                  unsigned int *seed_array)
{
    double sigma = 0.3;
    int max_length = 20;
    int bi1 = rand_r(&seed_array[tid]) % NUM_BEADS;
    int len = 1 + rand_r(&seed_array[tid]) % max_length;
    int bi2 = bi1 + len;

    if (bi2 >= NUM_BEADS) bi2 = NUM_BEADS - 1;

    int st = (bi1 > 0) ? bi1 : 1;
    int en = (bi2 < NUM_BEADS-1) ? bi2 + 1 : NUM_BEADS - 1;

    for (int i = st; i <= en; i++) {
        pos[snap_id][i][0] += sigma * rand_gaussian(&seed_array[tid]);
        pos[snap_id][i][1] += sigma * rand_gaussian(&seed_array[tid]);
        pos[snap_id][i][2] += sigma * rand_gaussian(&seed_array[tid]);
    }

    for (int i = st; i < NUM_BEADS; i++) {
        double ddx = pos[snap_id][i][0] - pos[snap_id][i-1][0];
        double ddy = pos[snap_id][i][1] - pos[snap_id][i-1][1];
        double ddz = pos[snap_id][i][2] - pos[snap_id][i-1][2];
        double dist = sqrt(ddx*ddx + ddy*ddy + ddz*ddz);

        if (dist < 1e-12) dist = 1e-12;

        double sc = bref[i-1] / dist;
        pos[snap_id][i][0] = pos[snap_id][i-1][0] + ddx*sc;
        pos[snap_id][i][1] = pos[snap_id][i-1][1] + ddy*sc;
        pos[snap_id][i][2] = pos[snap_id][i-1][2] + ddz*sc;
    }
}

static void pivot_move(int tid, int snap_id, double ***pos,
                       unsigned int *seed_array)
{
    int pp = rand_r(&seed_array[tid]) % NUM_BEADS;
    int S, E;

    if (pp < NUM_BEADS * 0.5) { S = 0;      E = pp; }
    else                      { S = pp + 1; E = NUM_BEADS; }

    if (E - S <= 1) return;

    double ux = 2.0*((double)rand_r(&seed_array[tid])/(double)RAND_MAX) - 1.0;
    double uy = 2.0*((double)rand_r(&seed_array[tid])/(double)RAND_MAX) - 1.0;
    double uz = 2.0*((double)rand_r(&seed_array[tid])/(double)RAND_MAX) - 1.0;
    double nm = sqrt(ux*ux + uy*uy + uz*uz);

    if (nm < 1e-12) return;

    ux /= nm; uy /= nm; uz /= nm;

    double angle = 2.0*M_PI*((double)rand_r(&seed_array[tid])/(double)RAND_MAX);
    double ca = cos(angle), sa = sin(angle), omc = 1.0 - ca;

    double R[3][3];
    R[0][0]=ca+ux*ux*omc;     R[0][1]=ux*uy*omc-uz*sa; R[0][2]=ux*uz*omc+uy*sa;
    R[1][0]=uy*ux*omc+uz*sa;  R[1][1]=ca+uy*uy*omc;    R[1][2]=uy*uz*omc-ux*sa;
    R[2][0]=uz*ux*omc-uy*sa;  R[2][1]=uz*uy*omc+ux*sa; R[2][2]=ca+uz*uz*omc;

    double ox = pos[snap_id][pp][0];
    double oy = pos[snap_id][pp][1];
    double oz = pos[snap_id][pp][2];

    for (int i = S; i < E; i++) {
        double x = pos[snap_id][i][0] - ox;
        double y = pos[snap_id][i][1] - oy;
        double z = pos[snap_id][i][2] - oz;

        pos[snap_id][i][0] = R[0][0]*x + R[0][1]*y + R[0][2]*z + ox;
        pos[snap_id][i][1] = R[1][0]*x + R[1][1]*y + R[1][2]*z + oy;
        pos[snap_id][i][2] = R[2][0]*x + R[2][1]*y + R[2][2]*z + oz;
    }
}

/* ══════════════════════ Energies / COM / bias ═══════════════════════ */
static double calculate_bond_energy(double ***pos, int lms)
{
    double e = 0.0;

    for (int i = 0; i < NUM_BEADS - 1; i++) {
        double dx = pos[lms][i+1][0] - pos[lms][i][0];
        double dy = pos[lms][i+1][1] - pos[lms][i][1];
        double dz = pos[lms][i+1][2] - pos[lms][i][2];
        double dr = sqrt(dx*dx + dy*dy + dz*dz) - bref[i];
        e += dr * dr;
    }

    return kharmonic * e;
}

static double calculate_bend_energy(double ***pos, int lms)
{
    double e = 0.0;

    for (int i = 0; i < NUM_ANGLE; i++) {
        e += angle_energy_triplet(pos[lms][P1[i]],
                                  pos[lms][P2[i]],
                                  pos[lms][P3[i]]);
    }

    return e;
}

static void update_all_segment_coms(int ms)
{
    int lms = LIDX(ms);

    for (int s = 0; s < NUM_SEGMENTS; s++) {
        double cx = 0.0, cy = 0.0, cz = 0.0;
        int st = segment_start[s];
        int en = segment_end[s];
        int cnt = en - st + 1;

        for (int b = st; b <= en; b++) {
            cx += positions[lms][b][0];
            cy += positions[lms][b][1];
            cz += positions[lms][b][2];
        }

        segment_com[lms][s][0] = cx / cnt;
        segment_com[lms][s][1] = cy / cnt;
        segment_com[lms][s][2] = cz / cnt;
    }
}

static void update_partial_segment_coms(int ms, int seg_lo, int seg_hi)
{
    int lms = LIDX(ms);

    for (int s = seg_lo; s <= seg_hi; s++) {
        double cx = 0.0, cy = 0.0, cz = 0.0;
        int st = segment_start[s];
        int en = segment_end[s];
        int cnt = en - st + 1;

        for (int b = st; b <= en; b++) {
            cx += positions[lms][b][0];
            cy += positions[lms][b][1];
            cz += positions[lms][b][2];
        }

        segment_com[lms][s][0] = cx / cnt;
        segment_com[lms][s][1] = cy / cnt;
        segment_com[lms][s][2] = cz / cnt;
    }
}

static double compute_raw_bias_from_coms(int ms)
{
    int lms = LIDX(ms);
    double en = 0.0;

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        for (int j = i + 1; j < NUM_SEGMENTS; j++) {
            if (expC[i][j] <= 1e-10) continue;

            double dx = segment_com[lms][i][0] - segment_com[lms][j][0];
            double dy = segment_com[lms][i][1] - segment_com[lms][j][1];
            double dz = segment_com[lms][i][2] - segment_com[lms][j][2];

            en += lamda[i][j] * contact_weight(sqrt(dx*dx + dy*dy + dz*dz));
        }
    }

    return en;
}

static void rebase_total_energy_from_components(int ms_s, int ms_e)
{
    for (int ms = ms_s; ms < ms_e; ms++) {
        int lms = LIDX(ms);
        old_energy[lms] = bond_energy_walker[lms]
                        + bend_energy_walker[lms]
                        + lj_energy_walker[lms]
                        + bias_beta * raw_bias_walker[lms];
    }
}

static void recompute_all_energies(int ms, int tid)
{
    int lms = LIDX(ms);

    update_all_segment_coms(ms);

    bond_energy_walker[lms] = calculate_bond_energy(positions, lms);
    bend_energy_walker[lms] = calculate_bend_energy(positions, lms);
    lj_energy_walker[lms]   = compute_lj_energy_bvh(positions[lms], NUM_BEADS, tid);
    raw_bias_walker[lms]    = compute_raw_bias_from_coms(ms);

    old_energy[lms] = bond_energy_walker[lms]
                    + bend_energy_walker[lms]
                    + lj_energy_walker[lms]
                    + bias_beta * raw_bias_walker[lms];
}

/* ══════════════════════ LJ pair kernels ═════════════════════════════ */
static inline int pair_is_excluded_topologically(int i1, int i2)
{
    if (abs(i2 - i1) <= 1) return 1;
    if (abs(i2 - i1) == 2 && btyp[i1] == 0 && btyp[i2] == 0) return 1;
    return 0;
}

static inline double pair_sigma(int i1, int i2)
{
    return rad[i1] + rad[i2];
}

static inline double pair_rcut_from_sigma(double sigma)
{
    return (fabs(sigma - 10.0) < 1e-12) ? 2.5*sigma : 1.1225*sigma;
}

static inline double pair_rcut(int i1, int i2)
{
    return pair_rcut_from_sigma(pair_sigma(i1, i2));
}

static inline int pair_can_interact(int i1, int i2, double r2)
{
    double rc = pair_rcut(i1, i2);
    return (r2 < rc * rc);
}

static inline double soft_repulsion_pair(int i1, int i2, double r2)
{
    double sigma = pair_sigma(i1, i2);
    double rcut  = pair_rcut_from_sigma(sigma);
    double sig2  = sigma * sigma;
    double rcut2 = rcut  * rcut;

    double r2ic  = sig2 / rcut2;
    double r6ic  = r2ic * r2ic * r2ic;
    double r12ic = r6ic * r6ic;

    double r2i   = sig2 / r2;
    double r6i   = r2i * r2i * r2i;
    double r12i  = r6i * r6i;

    return (r2 < rcut2) ? 0.1 * ((r12i - r6i) - (r12ic - r6ic)) : 0.0;
}

/* ══════════════════════ BVH helpers ═════════════════════════════════ */
static void compute_bounding_box(int *idx, int start, int count,
                                 double *bmin, double *bmax, double **pos)
{
    for (int d = 0; d < 3; ++d) {
        bmin[d] =  DBL_MAX;
        bmax[d] = -DBL_MAX;
    }

    for (int i = start; i < start + count; ++i) {
        int k = idx[i];
        for (int d = 0; d < 3; ++d) {
            double p = pos[k][d];
            double r = rad[k];
            if (p - r < bmin[d]) bmin[d] = p - r;
            if (p + r > bmax[d]) bmax[d] = p + r;
        }
    }
}

static int compare_indices_r(const void *a, const void *b, void *ctx)
{
    const SortCtx *s = (const SortCtx*)ctx;
    double ca = s->pos[*(const int*)a][s->axis];
    double cb = s->pos[*(const int*)b][s->axis];
    return (ca < cb) ? -1 : (ca > cb) ? 1 : 0;
}

static BVHNode *build_bvh_arena(int *idx, int start, int count,
                                double **pos, BVHArena *arena, SortCtx *ctx)
{
    if (arena->used >= BVH_POOL_SIZE) {
        fprintf(stderr, "Rank %d: BVH arena overflow\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    BVHNode *node = &arena->pool[arena->used++];
    node->start = start;
    node->count = count;
    node->left  = NULL;
    node->right = NULL;

    compute_bounding_box(idx, start, count, node->bbox_min, node->bbox_max, pos);

    if (count <= MAX_BEADS_IN_LEAF) {
        node->is_leaf = 1;
        return node;
    }

    node->is_leaf = 0;

    {
        int axis = 0;
        double maxlen = node->bbox_max[0] - node->bbox_min[0];

        for (int d = 1; d < 3; ++d) {
            double len = node->bbox_max[d] - node->bbox_min[d];
            if (len > maxlen) { maxlen = len; axis = d; }
        }

        ctx->axis = axis;
        qsort_r(idx + start, (size_t)count, sizeof(int), compare_indices_r, ctx);

        int lc  = count / 2;
        int mid = start + lc;

        node->left  = build_bvh_arena(idx, start, lc, pos, arena, ctx);
        node->right = build_bvh_arena(idx, mid, count - lc, pos, arena, ctx);
    }

    return node;
}

static BVHNode *build_wbvh_arena(int *idx, int start, int count,
                                 double **pos, WBVHArena *arena, SortCtx *ctx)
{
    if (arena->used >= WBVH_POOL_SIZE) {
        fprintf(stderr, "Rank %d: walker BVH arena overflow (used=%d)\n",
                mpi_rank, arena->used);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    BVHNode *node = &arena->pool[arena->used++];
    node->start = start;
    node->count = count;
    node->left  = NULL;
    node->right = NULL;

    compute_bounding_box(idx, start, count, node->bbox_min, node->bbox_max, pos);

    if (count <= MAX_BEADS_IN_LEAF) {
        node->is_leaf = 1;
        return node;
    }

    node->is_leaf = 0;

    {
        int axis = 0;
        double maxlen = node->bbox_max[0] - node->bbox_min[0];

        for (int d = 1; d < 3; ++d) {
            double len = node->bbox_max[d] - node->bbox_min[d];
            if (len > maxlen) { maxlen = len; axis = d; }
        }

        ctx->axis = axis;
        qsort_r(idx + start, (size_t)count, sizeof(int), compare_indices_r, ctx);

        int lc  = count / 2;
        int mid = start + lc;

        node->left  = build_wbvh_arena(idx, start, lc, pos, arena, ctx);
        node->right = build_wbvh_arena(idx, mid, count - lc, pos, arena, ctx);
    }

    return node;
}

static void bvh_query_sphere(BVHNode *node, int *idx, double **pos,
                             double cx, double cy, double cz, double qr,
                             int excl_lo, int excl_hi,
                             int *cands, int *ncands, char *flags)
{
    if (!sphere_box_overlap(cx, cy, cz, qr, node->bbox_min, node->bbox_max)) return;

    if (node->is_leaf) {
        int se = node->start;
        int ee = se + node->count;

        for (int a = se; a < ee; a++) {
            int j = idx[a];
            if (j >= excl_lo && j <= excl_hi) continue;
            if (flags[j]) continue;

            double dx = pos[j][0] - cx;
            double dy = pos[j][1] - cy;
            double dz = pos[j][2] - cz;

            if (dx*dx + dy*dy + dz*dz <= qr*qr) {
                flags[j] = 1;
                cands[(*ncands)++] = j;
            }
        }
        return;
    }

    bvh_query_sphere(node->left,  idx, pos, cx, cy, cz, qr, excl_lo, excl_hi,
                     cands, ncands, flags);
    bvh_query_sphere(node->right, idx, pos, cx, cy, cz, qr, excl_lo, excl_hi,
                     cands, ncands, flags);
}

static void rebuild_walker_bvh(int ms)
{
    int local_idx = LIDX(ms);
    WBVHArena *arena = &walker_wbvh[local_idx];
    int *idx = walker_wbvh_idx[local_idx];

    arena->used = 0;
    for (int i = 0; i < NUM_BEADS; i++) idx[i] = i;

    SortCtx ctx;
    ctx.pos  = positions[local_idx];
    ctx.axis = 0;

    walker_wbvh_roots[local_idx] = build_wbvh_arena(idx, 0, NUM_BEADS,
                                                    positions[local_idx], arena, &ctx);
}

static inline int boxes_overlap(const double *mn1, const double *mx1,
                                const double *mn2, const double *mx2, double m)
{
    for (int d = 0; d < 3; ++d)
        if (mn1[d] > mx2[d] + m || mx1[d] < mn2[d] - m) return 0;
    return 1;
}

static void process_leaf_self(BVHNode *leaf, double *e, double **pos, int *idx)
{
    int se = leaf->start;
    int ee = se + leaf->count;

    for (int a = se; a < ee; ++a) {
        int i = idx[a];
        for (int b = a + 1; b < ee; ++b) {
            int j = idx[b];
            if (pair_is_excluded_topologically(i, j)) continue;

            double dx = pos[i][0] - pos[j][0];
            double dy = pos[i][1] - pos[j][1];
            double dz = pos[i][2] - pos[j][2];
            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < 1e-12) r2 = 1e-12;

            if (pair_can_interact(i, j, r2))
                *e += soft_repulsion_pair(i, j, r2);
        }
    }
}

static void process_leaf_cross(BVHNode *a, BVHNode *b, double *e, double **pos, int *idx)
{
    int as = a->start, ae = as + a->count;
    int bs = b->start, be = bs + b->count;

    for (int ia = as; ia < ae; ++ia) {
        int i = idx[ia];
        for (int ib = bs; ib < be; ++ib) {
            int j = idx[ib];
            if (pair_is_excluded_topologically(i, j)) continue;

            double dx = pos[i][0] - pos[j][0];
            double dy = pos[i][1] - pos[j][1];
            double dz = pos[i][2] - pos[j][2];
            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < 1e-12) r2 = 1e-12;

            if (pair_can_interact(i, j, r2))
                *e += soft_repulsion_pair(i, j, r2);
        }
    }
}

static void traverse_bvh_cross(BVHNode *a, BVHNode *b, double *e, double **pos, int *idx)
{
    if (!boxes_overlap(a->bbox_min, a->bbox_max, b->bbox_min, b->bbox_max, MAX_RCUT))
        return;

    if (a->is_leaf && b->is_leaf) {
        process_leaf_cross(a, b, e, pos, idx);
        return;
    }

    if (a->is_leaf) {
        traverse_bvh_cross(a, b->left,  e, pos, idx);
        traverse_bvh_cross(a, b->right, e, pos, idx);
    } else if (b->is_leaf) {
        traverse_bvh_cross(a->left,  b, e, pos, idx);
        traverse_bvh_cross(a->right, b, e, pos, idx);
    } else if (a->count >= b->count) {
        traverse_bvh_cross(a->left,  b, e, pos, idx);
        traverse_bvh_cross(a->right, b, e, pos, idx);
    } else {
        traverse_bvh_cross(a, b->left,  e, pos, idx);
        traverse_bvh_cross(a, b->right, e, pos, idx);
    }
}

static void traverse_bvh_self(BVHNode *node, double *e, double **pos, int *idx)
{
    if (node->is_leaf) {
        process_leaf_self(node, e, pos, idx);
        return;
    }

    traverse_bvh_self(node->left, e, pos, idx);
    traverse_bvh_cross(node->left, node->right, e, pos, idx);
    traverse_bvh_self(node->right, e, pos, idx);
}

static double compute_lj_energy_bvh(double **pos, int num_beads, int tid)
{
    BVHArena *arena = &bvh_arenas[tid];
    int *idx = bvh_idx_bufs[tid];

    arena->used = 0;
    for (int i = 0; i < num_beads; ++i) idx[i] = i;

    SortCtx ctx;
    ctx.pos  = pos;
    ctx.axis = 0;

    BVHNode *root = build_bvh_arena(idx, 0, num_beads, pos, arena, &ctx);

    double e = 0.0;
    traverse_bvh_self(root, &e, pos, idx);
    return e;
}

/* ══════════════════════ Contact map calculation only ═══════════════ */

static void calculate_contact_map_mpi(int ms_s, int ms_e, int update_lambda)
{
    int nt  = omp_get_max_threads();
    int NS  = NUM_SEGMENTS;
    size_t NS2 = (size_t)NS * (size_t)NS;
    int Npairs = (NS * (NS - 1)) / 2;

    memset(g_thread_maps_flat, 0, (size_t)nt * NS2 * sizeof(double));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *lc = g_thread_maps_flat + (size_t)tid * NS2;

#pragma omp for schedule(static)
        for (int ms = ms_s; ms < ms_e; ms++) {
            int lms = LIDX(ms);

            for (int ii = 0; ii < NS; ii++) {
                for (int jj = ii + 1; jj < NS; jj++) {
                    double dx = segment_com[lms][ii][0] - segment_com[lms][jj][0];
                    double dy = segment_com[lms][ii][1] - segment_com[lms][jj][1];
                    double dz = segment_com[lms][ii][2] - segment_com[lms][jj][2];
                    double w  = contact_weight(sqrt(dx*dx + dy*dy + dz*dz));

                    lc[ii * NS + jj] += w;
                    lc[jj * NS + ii] += w;
                }
            }
        }
    }

    memset(g_allreduce_local, 0, NS2 * sizeof(double));
    for (int t = 0; t < nt; t++) {
        double *tm = g_thread_maps_flat + (size_t)t * NS2;
        for (size_t k = 0; k < NS2; k++)
            g_allreduce_local[k] += tm[k];
    }

    MPI_Allreduce(g_allreduce_local, g_allreduce_global, (int)NS2,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < NS; i++) {
        for (int j = 0; j < NS; j++) {
            prevSimC[i][j] = simC[i][j];
        }
    }

    {
        double inv_tsnap = 1.0 / (double)Tsnap;
        for (int i = 0; i < NS; i++) {
            for (int j = 0; j < NS; j++) {
                simC[i][j] = g_allreduce_global[(size_t)i * NS + j] * inv_tsnap;
            }
        }
    }

    if (update_lambda) {
        for (int i = 0; i < NS; i++) {
            lamda[i][i] = 0.0;

            for (int j = i + 1; j < NS; j++) {
                if (expC[i][j] <= 1e-12) {
                    lamda[i][j] = 0.0;
                    lamda[j][i] = 0.0;
                    continue;
                }

                double dij = (double)(j - i);
                double gdist = log(dij * dij + 1.0);
                double strength_factor = 1.0 + expC[i][j];
                double log_ratio = log((simC[i][j] + 1e-6) / (expC[i][j] + 1e-6));
                double delta = etadiff * gdist * strength_factor * log_ratio;

                lamda[i][j] += delta;
                lamda[j][i]  = lamda[i][j];
            }
        }

        for (int ms = ms_s; ms < ms_e; ms++) {
            int lms = LIDX(ms);
            raw_bias_walker[lms] = compute_raw_bias_from_coms(ms);
            check_energy_finite(raw_bias_walker[lms], "raw_bias_after_lambda_update", -1, ms);
        }
        rebase_total_energy_from_components(ms_s, ms_e);
    }

    if (mpi_rank == 0) {
        double total_err = 0.0;
        for (int i = 0; i < NS - 1; i++) {
            for (int j = i + 1; j < NS; j++) {
                double diff = simC[i][j] - expC[i][j];
                total_err += diff * diff;
            }
        }
        double rms = sqrt(total_err / (double)Npairs);
        printf("RMS error = %.8f\n", rms);
    }
}

/* ══════════════════════ Initialisation ══════════════════════════════ */
static void init(void)
{
    local_Tsnap = ms_end - ms_start;
    srand((unsigned int)time(NULL) + (unsigned int)mpi_rank * 104729u);

    int i, j;
    double x, y, z;

    positions     = (double***)calloc((size_t)local_Tsnap, sizeof(double**));
old_positions = (double***)calloc((size_t)local_Tsnap, sizeof(double**));
    rad           = (double*)calloc(NUM_BEADS, sizeof(double));
    bref          = (double*)calloc(NUM_BEADS, sizeof(double));
    btyp          = (int*)calloc(NUM_BEADS, sizeof(int));
    P1            = (int*)calloc(NUM_ANGLE, sizeof(int));
    P2            = (int*)calloc(NUM_ANGLE, sizeof(int));
    P3            = (int*)calloc(NUM_ANGLE, sizeof(int));

    old_energy         = (double*)calloc((size_t)local_Tsnap, sizeof(double));
    new_energy         = (double*)calloc((size_t)local_Tsnap, sizeof(double));
    lj_energy_walker   = (double*)calloc((size_t)local_Tsnap, sizeof(double));
    bond_energy_walker = (double*)calloc((size_t)local_Tsnap, sizeof(double));
    bend_energy_walker = (double*)calloc((size_t)local_Tsnap, sizeof(double));
    raw_bias_walker    = (double*)calloc((size_t)local_Tsnap, sizeof(double));

    if (!positions || !old_positions || !rad || !bref || !btyp ||
        !P1 || !P2 || !P3 || !old_energy || !new_energy ||
        !lj_energy_walker || !bond_energy_walker ||
        !bend_energy_walker || !raw_bias_walker) {
        fprintf(stderr, "Rank %d: core allocation failed\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (j = 0; j < local_Tsnap; j++) { 
        positions[j]     = (double**)calloc(NUM_BEADS, sizeof(double*));
        old_positions[j] = (double**)calloc(NUM_BEADS, sizeof(double*));
        if (!positions[j] || !old_positions[j]) {
            fprintf(stderr, "Rank %d: snapshot allocation failed at %d\n", mpi_rank, j);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (i = 0; i < NUM_BEADS; i++) {
            positions[j][i]     = (double*)calloc(3, sizeof(double));
            old_positions[j][i] = (double*)calloc(3, sizeof(double));
            if (!positions[j][i] || !old_positions[j][i]) {
                fprintf(stderr, "Rank %d: bead allocation failed at snap %d bead %d\n",
                        mpi_rank, j, i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    expC  = (double**)calloc(NUM_SEGMENTS, sizeof(double*));
    simC  = (double**)calloc(NUM_SEGMENTS, sizeof(double*));
    lamda = (double**)calloc(NUM_SEGMENTS, sizeof(double*));
    prevSimC = (double**)calloc(NUM_SEGMENTS, sizeof(double*));

    if (!expC || !simC || !lamda || !prevSimC) {
        fprintf(stderr, "Rank %d: segment matrix allocation failed\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (i = 0; i < NUM_SEGMENTS; i++) {
        expC[i]    = (double*)calloc(NUM_SEGMENTS, sizeof(double));
        simC[i]    = (double*)calloc(NUM_SEGMENTS, sizeof(double));
        prevSimC[i]= (double*)calloc(NUM_SEGMENTS, sizeof(double));
        lamda[i]   = (double*)calloc(NUM_SEGMENTS, sizeof(double));

        if (!expC[i] || !simC[i] || !prevSimC[i] || !lamda[i]) {
            fprintf(stderr, "Rank %d: segment row allocation failed at %d\n", mpi_rank, i);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    segment_com = (double***)calloc((size_t)local_Tsnap, sizeof(double**));
    if (!segment_com) {
        fprintf(stderr, "Rank %d: segment_com allocation failed\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int t = 0; t < local_Tsnap; t++) {
        segment_com[t] = (double**)calloc(NUM_SEGMENTS, sizeof(double*));
        if (!segment_com[t]) {
            fprintf(stderr, "Rank %d: segment_com[%d] allocation failed\n", mpi_rank, t);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int s = 0; s < NUM_SEGMENTS; s++) {
            segment_com[t][s] = (double*)calloc(3, sizeof(double));
            if (!segment_com[t][s]) {
                fprintf(stderr, "Rank %d: segment_com[%d][%d] allocation failed\n",
                        mpi_rank, t, s);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    /* per-thread transient BVH arenas */
    int max_threads = omp_get_max_threads();

    bvh_arenas   = (BVHArena*)malloc((size_t)max_threads * sizeof(BVHArena));
    bvh_idx_bufs = (int**)malloc((size_t)max_threads * sizeof(int*));

    if (!bvh_arenas || !bvh_idx_bufs) {
        fprintf(stderr, "Rank %d: BVH thread allocations failed\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int t = 0; t < max_threads; t++) {
        bvh_arenas[t].used = 0;
        bvh_idx_bufs[t] = (int*)malloc((size_t)NUM_BEADS * sizeof(int));
        if (!bvh_idx_bufs[t]) {
            fprintf(stderr, "Rank %d: BVH idx buffer allocation failed for thread %d\n",
                    mpi_rank, t);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* contact-map buffers */
    g_max_threads_cm   = max_threads;
    {
        size_t NS2 = (size_t)NUM_SEGMENTS * (size_t)NUM_SEGMENTS;
        g_thread_maps_flat = (double*)malloc((size_t)max_threads * NS2 * sizeof(double));
        g_allreduce_local  = (double*)malloc(NS2 * sizeof(double));
        g_allreduce_global = (double*)malloc(NS2 * sizeof(double));

        if (!g_thread_maps_flat || !g_allreduce_local || !g_allreduce_global) {
            fprintf(stderr, "Rank %d: contact-map buffer allocation failed\n", mpi_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* persistent walker BVH */
    {
        int local_count = ms_end - ms_start;
        walker_wbvh       = (WBVHArena*)malloc((size_t)local_count * sizeof(WBVHArena));
        walker_wbvh_idx   = (int**)malloc((size_t)local_count * sizeof(int*));
        walker_wbvh_roots = (BVHNode**)malloc((size_t)local_count * sizeof(BVHNode*));

        if (!walker_wbvh || !walker_wbvh_idx || !walker_wbvh_roots) {
            fprintf(stderr, "Rank %d: walker BVH allocation failed\n", mpi_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int li = 0; li < local_count; li++) {
            walker_wbvh[li].used = 0;
            walker_wbvh_roots[li] = NULL;
            walker_wbvh_idx[li] = (int*)malloc((size_t)NUM_BEADS * sizeof(int));
            if (!walker_wbvh_idx[li]) {
                fprintf(stderr, "Rank %d: walker BVH idx allocation failed for local walker %d\n",
                        mpi_rank, li);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    /* candidate buffers */
    thread_cand_buf   = (int**)malloc((size_t)max_threads * sizeof(int*));
    thread_cand_flags = (char**)malloc((size_t)max_threads * sizeof(char*));

    if (!thread_cand_buf || !thread_cand_flags) {
        fprintf(stderr, "Rank %d: candidate buffer allocation failed\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int t = 0; t < max_threads; t++) {
        thread_cand_buf[t]   = (int*)malloc((size_t)NUM_BEADS * sizeof(int));
        thread_cand_flags[t] = (char*)calloc(NUM_BEADS, sizeof(char));
        if (!thread_cand_buf[t] || !thread_cand_flags[t]) {
            fprintf(stderr, "Rank %d: candidate buffer allocation failed for thread %d\n",
                    mpi_rank, t);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* read coarse graining map */
    segment_sizes = (int*)calloc(NUM_SEGMENTS, sizeof(int));
    segment_start = (int*)calloc(NUM_SEGMENTS, sizeof(int));
    segment_end   = (int*)calloc(NUM_SEGMENTS, sizeof(int));
    if (!segment_sizes || !segment_start || !segment_end) {
        fprintf(stderr, "Rank %d: segment metadata allocation failed\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    {
        FILE *fp = fopen(NL2CG, "r");
        if (!fp) {
            fprintf(stderr, "Rank %d: cannot open NL2CG\n", mpi_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int s = 0; s < NUM_SEGMENTS; s++) {
            if (fscanf(fp, "%d", &segment_sizes[s]) != 1) {
                fprintf(stderr, "Rank %d: NL2CG read error at segment %d\n", mpi_rank, s);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        fclose(fp);
    }

    segment_start[0] = 0;
    segment_end[0]   = segment_sizes[0] - 1;
    for (int s = 1; s < NUM_SEGMENTS; s++) {
        segment_start[s] = segment_end[s-1] + 1;
        segment_end[s]   = segment_start[s] + segment_sizes[s] - 1;
    }

    if (segment_end[NUM_SEGMENTS - 1] != NUM_BEADS - 1) {
        fprintf(stderr,
                "Rank %d: NL2CG inconsistent with NUM_BEADS (last end=%d, expected=%d)\n",
                mpi_rank, segment_end[NUM_SEGMENTS - 1], NUM_BEADS - 1);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    seg_for_bead = (int*)malloc((size_t)NUM_BEADS * sizeof(int));
    if (!seg_for_bead) {
        fprintf(stderr, "Rank %d: seg_for_bead allocation failed\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int s = 0; s < NUM_SEGMENTS; s++)
        for (int b = segment_start[s]; b <= segment_end[s]; b++)
            seg_for_bead[b] = s;

    /* angle inverse index */
    bead_angle_count = (int*)calloc(NUM_BEADS, sizeof(int));
    bead_to_angles   = (int**)malloc((size_t)NUM_BEADS * sizeof(int*));
    if (!bead_angle_count || !bead_to_angles) {
        fprintf(stderr, "Rank %d: angle inverse index allocation failed\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* bead types and radii */
    {
        FILE *nucpos = fopen(NUCPOS, "r");
        if (!nucpos) {
            fprintf(stderr, "Rank %d: cannot open NUCPOS\n", mpi_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (i = 0; i < NUM_BEADS; i++) {
            if (fscanf(nucpos, "%d", &btyp[i]) != 1) {
                fprintf(stderr, "Rank %d: NUCPOS read error at bead %d\n", mpi_rank, i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            rad[i] = (btyp[i] == 0) ? 1.25 : 5.0;
        }
        fclose(nucpos);
    }

    for (i = 0; i < NUM_BEADS - 1; i++)
        bref[i] = rad[i] + rad[i+1];

    {
        FILE *angle = fopen(ANGINFO, "r");
        if (!angle) {
            fprintf(stderr, "Rank %d: cannot open ANGINFO\n", mpi_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (i = 0; i < NUM_ANGLE; i++) {
            if (fscanf(angle, "%d %d %d", &P1[i], &P2[i], &P3[i]) != 3) {
                fprintf(stderr, "Rank %d: ANGINFO read error at line %d\n", mpi_rank, i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            bead_angle_count[P1[i]]++;
            bead_angle_count[P2[i]]++;
            bead_angle_count[P3[i]]++;
        }
        fclose(angle);
    }

    for (int b = 0; b < NUM_BEADS; b++) {
        bead_to_angles[b] = (int*)malloc((size_t)bead_angle_count[b] * sizeof(int));
        if (bead_angle_count[b] > 0 && !bead_to_angles[b]) {
            fprintf(stderr, "Rank %d: bead_to_angles allocation failed for bead %d\n",
                    mpi_rank, b);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    {
        int *fill = (int*)calloc(NUM_BEADS, sizeof(int));
        if (!fill) {
            fprintf(stderr, "Rank %d: fill allocation failed\n", mpi_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (i = 0; i < NUM_ANGLE; i++) {
            int b;
            b = P1[i]; bead_to_angles[b][fill[b]++] = i;
            b = P2[i]; bead_to_angles[b][fill[b]++] = i;
            b = P3[i]; bead_to_angles[b][fill[b]++] = i;
        }
        free(fill);
    }

    /* experimental contact map + fixed lambda */
        /* experimental contact map + initial lambda = 0 */
    {
        FILE *expCont = fopen(EXPC, "r");
        if (!expCont) {
            fprintf(stderr, "Rank %d: cannot open EXPC\n", mpi_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        double max_raw = 0.0;
        int idx1, idx2;

        for (i = 0; i < NUM_SEGMENTS; i++) {
            for (j = 0; j < NUM_SEGMENTS; j++) {
                if (fscanf(expCont, "%d %d %lf", &idx1, &idx2, &expC[i][j]) != 3) {
                    fprintf(stderr, "Rank %d: EXPC read error at (%d,%d)\n", mpi_rank, i, j);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (idx1 != i || idx2 != j) {
                    fprintf(stderr,
                            "Rank %d: EXPC index mismatch at (%d,%d), file says (%d,%d)\n",
                            mpi_rank, i, j, idx1, idx2);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                /* backward MaxEnt: start from lambda = 0 */
                lamda[i][j] = 0.0;

                if (expC[i][j] == -1.0 || i == j)
                    expC[i][j] = 0.0;
                else if (expC[i][j] > max_raw)
                    max_raw = expC[i][j];
            }
        }

        fclose(expCont);

        {
            double C0 = 0.2 * max_raw;
            for (i = 0; i < NUM_SEGMENTS; i++) {
                for (j = 0; j < NUM_SEGMENTS; j++) {
                    double C = expC[i][j];
                    expC[i][j] = (C <= 0.0) ? 0.0 : C / (C + C0);
                }
            }
        }

        {
            double freqmax = 0.0;
            if (mpi_rank == 0) {
                for (i = 0; i < NUM_SEGMENTS - 1; i++)
                    freqmax += expC[i][i+1];
                if (freqmax > 0.0)
                    freqmax = (double)(NUM_SEGMENTS - 1) / freqmax;
            }

            MPI_Bcast(&freqmax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            for (i = 0; i < NUM_SEGMENTS; i++) {
                for (j = 0; j < NUM_SEGMENTS; j++) {
                    if (i == j) expC[i][j] = 0.0;
                    expC[i][j] *= freqmax;
                }
            }

            if (mpi_rank == 0) {
                FILE *ecm = fopen(ECM, "w");
                if (!ecm) {
                    fprintf(stderr, "Cannot open %s\n", ECM);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                for (i = 0; i < NUM_SEGMENTS; i++) {
                    for (j = 0; j < NUM_SEGMENTS; j++)
                        fprintf(ecm, "%d %d %f\n", i, j, expC[i][j]);
                    fprintf(ecm, "\n");
                }
                fclose(ecm);
            }
        }
    }
    /* initial conformations */
    {
    FILE *initS = fopen(IENSEM, "r");
    if (!initS) {
        fprintf(stderr, "Rank %d: cannot open IENSEM\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int ms = 0; ms < ms_start; ms++) {
        for (i = 0; i < NUM_BEADS; i++) {
            if (fscanf(initS, "%lf %lf %lf", &x, &y, &z) != 3) {
                fprintf(stderr, "Rank %d: IENSEM skip read error at walker %d bead %d\n",
                        mpi_rank, ms, i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    for (int ms = ms_start; ms < ms_end; ms++) {
        int lms = LIDX(ms);
        for (i = 0; i < NUM_BEADS; i++) {
            if (fscanf(initS, "%lf %lf %lf", &x, &y, &z) != 3) {
                fprintf(stderr, "Rank %d: IENSEM read error at walker %d bead %d\n",
                        mpi_rank, ms, i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            positions[lms][i][0] = x;
            positions[lms][i][1] = y;
            positions[lms][i][2] = z;
        }
    }

    fclose(initS);
}

    /* initial energies and walker BVHs for local walkers */
    for (int ms = ms_start; ms < ms_end; ms++) {
        recompute_all_energies(ms, 0);
        rebuild_walker_bvh(ms);
    }

    if (mpi_rank == 0) {
        printf("Rank 0 init complete: Tsnap=%d, local=[%d,%d)\n"
               "Walker BVH pool/walker=%d nodes (%zu KB)\n",
               Tsnap, ms_start, ms_end,
               WBVH_POOL_SIZE, sizeof(WBVHArena)/1024);
    }
}



/* ══════════════════════════ cleanup function ════════════════════════════════════ */


static void cleanup(void)
{
    int i, j;

    /* ── positions & old_positions ───────────────────────────── */
    if (positions) {
        for (i = 0; i < local_Tsnap; i++) {
            if (positions[i]) {
                for (j = 0; j < NUM_BEADS; j++) {
                    free(positions[i][j]);
                }
                free(positions[i]);
            }
        }
        free(positions);
    }

    if (old_positions) {
        for (i = 0; i < local_Tsnap; i++) {
            if (old_positions[i]) {
                for (j = 0; j < NUM_BEADS; j++) {
                    free(old_positions[i][j]);
                }
                free(old_positions[i]);
            }
        }
        free(old_positions);
    }

    /* ── segment COM ─────────────────────────────────────────── */
    if (segment_com) {
        for (i = 0; i < local_Tsnap; i++) {
            if (segment_com[i]) {
                for (j = 0; j < NUM_SEGMENTS; j++) {
                    free(segment_com[i][j]);
                }
                free(segment_com[i]);
            }
        }
        free(segment_com);
    }

    /* ── segment matrices ────────────────────────────────────── */
    if (expC) {
        for (i = 0; i < NUM_SEGMENTS; i++) free(expC[i]);
        free(expC);
    }

    if (simC) {
        for (i = 0; i < NUM_SEGMENTS; i++) free(simC[i]);
        free(simC);
    }

    if (lamda) {
        for (i = 0; i < NUM_SEGMENTS; i++) free(lamda[i]);
        free(lamda);
    }
    if (prevSimC) {
      for (i = 0; i < NUM_SEGMENTS; i++) free(prevSimC[i]);
      free(prevSimC);
    }

    /* ── per-bead arrays ─────────────────────────────────────── */
    free(rad);
    free(bref);
    free(btyp);
    free(P1);
    free(P2);
    free(P3);

    /* ── energies ───────────────────────────────────────────── */
    free(old_energy);
    free(new_energy);
    free(lj_energy_walker);
    free(bond_energy_walker);
    free(bend_energy_walker);
    free(raw_bias_walker);

    /* ── segment metadata ───────────────────────────────────── */
    free(segment_sizes);
    free(segment_start);
    free(segment_end);
    free(seg_for_bead);

    /* ── bead → angle mapping ───────────────────────────────── */
    if (bead_to_angles) {
        for (i = 0; i < NUM_BEADS; i++) {
            free(bead_to_angles[i]);
        }
        free(bead_to_angles);
    }
    free(bead_angle_count);

    /* ── BVH thread buffers ─────────────────────────────────── */
    if (bvh_idx_bufs) {
        int nt = omp_get_max_threads();
        for (i = 0; i < nt; i++) {
            free(bvh_idx_bufs[i]);
        }
        free(bvh_idx_bufs);
    }

    free(bvh_arenas);

    /* ── walker BVH ─────────────────────────────────────────── */
    if (walker_wbvh_idx) {
        int local_count = ms_end - ms_start;
        for (i = 0; i < local_count; i++) {
            free(walker_wbvh_idx[i]);
        }
        free(walker_wbvh_idx);
    }

    free(walker_wbvh);
    free(walker_wbvh_roots);

    /* ── candidate buffers ──────────────────────────────────── */
    if (thread_cand_buf) {
        int nt = omp_get_max_threads();
        for (i = 0; i < nt; i++) {
            free(thread_cand_buf[i]);
        }
        free(thread_cand_buf);
    }

    if (thread_cand_flags) {
        int nt = omp_get_max_threads();
        for (i = 0; i < nt; i++) {
            free(thread_cand_flags[i]);
        }
        free(thread_cand_flags);
    }

    /* ── contact-map buffers ───────────────────────────────── */
    free(g_thread_maps_flat);
    free(g_allreduce_local);
    free(g_allreduce_global);
}
/* ══════════════════════════ main ════════════════════════════════════ */
/* ══════════════════════════ main ════════════════════════════════════ */
int main(int argc, char **argv)
{
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (mpi_rank == 0)
            fprintf(stderr, "MPI does not provide required threading level\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

#ifdef _OPENMP
    omp_set_dynamic(0);
#endif

    ms_start    = (Tsnap * mpi_rank) / mpi_size;
    ms_end      = (Tsnap * (mpi_rank + 1)) / mpi_size;
    local_Tsnap = ms_end - ms_start;

    if (mpi_rank == 0) {
        printf("MPI size=%d  Tsnap=%d  local_Tsnap(avg)≈%d  MAX_MOVES=%d\n"
               "Schedule: %.0f%% crankshaft / %.0f%% gaussian / %.0f%% pivot\n",
               mpi_size, Tsnap, Tsnap / mpi_size, MAX_MOVES,
               (1.0 - probB) * 100.0,
               (probB - probA) * 100.0,
               probA * 100.0);
    }

    init();

    if (mpi_rank == 0) {
        printf("======= INIT COMPLETE =======\n"
               "Tsnap=%d NUM_BEADS=%d NUM_SEGMENTS=%d NUM_ANGLE=%d\n",
               Tsnap, NUM_BEADS, NUM_SEGMENTS, NUM_ANGLE);
    }

    /* per-thread RNG seeds */
    int max_threads = omp_get_max_threads();
    unsigned int *seed_array =
        (unsigned int *)malloc((size_t)max_threads * sizeof(unsigned int));

    if (!seed_array) {
        fprintf(stderr, "Rank %d: failed to allocate seed_array\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    {
        unsigned int base = (unsigned int)time(NULL);
        for (int t = 0; t < max_threads; t++) {
            unsigned int gtid = (unsigned int)(mpi_rank * max_threads + t);
            seed_array[t] = base ^ (gtid * 7919u);
            if (seed_array[t] == 0u) seed_array[t] = 1234567u + gtid;
        }
    }

    /* ═════════════════ Main MC loop ═══════════════════════════════ */
    for (int move = 0; move < MAX_MOVES; move++) {

        int accepted_count_local = 0;

#pragma omp parallel reduction(+:accepted_count_local)
        {
            int tid = omp_get_thread_num();

#pragma omp for schedule(static)
            for (int ms = ms_start; ms < ms_end; ms++) {

                int lms = LIDX(ms);

                /* periodic exact refresh to control drift */
                if (move > 0 && (move % REBASE_INTERVAL) == 0) {
                    recompute_all_energies(ms, tid);
                    check_positions_finite(ms, "recompute_all_energies", move);
                    check_energy_finite(old_energy[lms], "old_energy", move, ms);
                }

                double saved_bond = bond_energy_walker[lms];
                double saved_bend = bend_energy_walker[lms];
                double saved_lj   = lj_energy_walker[lms];
                double saved_rb   = raw_bias_walker[lms];

                double ran = (double)rand_r(&seed_array[tid]) /
                             ((double)RAND_MAX + 1.0);

                /* ========================================================
 *                  * CRANKSHAFT branch
 *                                   * ======================================================== */
                if (ran >= probB) {

                    int max_crank_len = 20;
                    int bead1 = rand_r(&seed_array[tid]) % NUM_BEADS;
                    int clen  = 2 + rand_r(&seed_array[tid]) % (max_crank_len - 1);
                    int bead2 = bead1 + clen;
                    if (bead2 >= NUM_BEADS) bead2 = NUM_BEADS - 1;
                    if (bead2 - bead1 < 2) goto NEXT_WALKER;

                    int b1 = bead1, b2 = bead2;
                    int mov_s = bead1 + 1, mov_e = bead2 - 1;
                    int n_moved = mov_e - mov_s + 1;
                    if (n_moved <= 0 || n_moved > MAX_CRANK_BEADS) goto NEXT_WALKER;

                    double ax = positions[lms][b2][0] - positions[lms][b1][0];
                    double ay = positions[lms][b2][1] - positions[lms][b1][1];
                    double az = positions[lms][b2][2] - positions[lms][b1][2];
                    double ar = sqrt(ax*ax + ay*ay + az*az);
                    if (ar < 1e-8) goto NEXT_WALKER;

                    double nx = ax / ar, ny = ay / ar, nz = az / ar;
                    double cang = 2.0 * M_PI *
                                  ((double)rand_r(&seed_array[tid]) / ((double)RAND_MAX + 1.0));
                    double ca = cos(cang), sa = sin(cang), omc = 1.0 - ca;

                    double RM[3][3];
                    RM[0][0] = ca + nx*nx*omc;    RM[0][1] = nx*ny*omc - nz*sa; RM[0][2] = nx*nz*omc + ny*sa;
                    RM[1][0] = ny*nx*omc + nz*sa; RM[1][1] = ca + ny*ny*omc;    RM[1][2] = ny*nz*omc - nx*sa;
                    RM[2][0] = nz*nx*omc - ny*sa; RM[2][1] = nz*ny*omc + nx*sa; RM[2][2] = ca + nz*nz*omc;

                    double ox = positions[lms][b1][0];
                    double oy = positions[lms][b1][1];
                    double oz = positions[lms][b1][2];

                    double old_p[MAX_CRANK_BEADS][3];
                    for (int k = 0; k < n_moved; k++) {
                        old_p[k][0] = positions[lms][mov_s + k][0];
                        old_p[k][1] = positions[lms][mov_s + k][1];
                        old_p[k][2] = positions[lms][mov_s + k][2];
                    }

                    double old_bl = 0.0, old_br = 0.0;
                    {
                        double *pf = positions[lms][b1];
                        double *pm = positions[lms][mov_s];
                        double ddx = pf[0] - pm[0];
                        double ddy = pf[1] - pm[1];
                        double ddz = pf[2] - pm[2];
                        double dr  = sqrt(ddx*ddx + ddy*ddy + ddz*ddz) - bref[b1];
                        old_bl = kharmonic * dr * dr;
                    }
                    if (b2 < NUM_BEADS) {
                        double *pf = positions[lms][b2];
                        double *pm = positions[lms][mov_e];
                        double ddx = pm[0] - pf[0];
                        double ddy = pm[1] - pf[1];
                        double ddz = pm[2] - pf[2];
                        double dr  = sqrt(ddx*ddx + ddy*ddy + ddz*ddz) - bref[mov_e];
                        old_br = kharmonic * dr * dr;
                    }

                    int ba_list[128];
                    int n_ba = 0;
                    {
                        int vis[128];
                        int nvis = 0;
                        int bnds[2] = {b1, b2};

                        for (int bi = 0; bi < 2; bi++) {
                            int bnd = bnds[bi];
                            if (bnd < 0 || bnd >= NUM_BEADS) continue;

                            for (int k = 0; k < bead_angle_count[bnd]; k++) {
                                int ai = bead_to_angles[bnd][k];
                                int dup = 0;
                                for (int vv = 0; vv < nvis; vv++) {
                                    if (vis[vv] == ai) { dup = 1; break; }
                                }
                                if (dup) continue;
                                vis[nvis++] = ai;

                                {
                                    int i1 = P1[ai], i2 = P2[ai], i3 = P3[ai];
                                    if (!((i1 >= mov_s && i1 <= mov_e) ||
                                          (i2 >= mov_s && i2 <= mov_e) ||
                                          (i3 >= mov_s && i3 <= mov_e))) continue;
                                }

                                if (n_ba < 128) ba_list[n_ba++] = ai;
                            }
                        }
                    }

                    double old_bend_bd = 0.0;
                    for (int k = 0; k < n_ba; k++) {
                        old_bend_bd += angle_energy_triplet(
                            positions[lms][P1[ba_list[k]]],
                            positions[lms][P2[ba_list[k]]],
                            positions[lms][P3[ba_list[k]]]
                        );
                    }

                    char *flags = thread_cand_flags[tid];
                    int  *cands = thread_cand_buf[tid];
                    int   ncands = 0;
                    memset(flags, 0, (size_t)NUM_BEADS * sizeof(char));

                    {
                        int local_idx = lms;
                        BVHNode *wroot = walker_wbvh_roots[local_idx];
                        int     *widx  = walker_wbvh_idx[local_idx];

                        for (int k = 0; k < n_moved; k++) {
                            bvh_query_sphere(wroot, widx, positions[lms],
                                             old_p[k][0], old_p[k][1], old_p[k][2],
                                             MAX_RCUT, mov_s, mov_e,
                                             cands, &ncands, flags);
                        }

                        int seg_lo = seg_for_bead[mov_s];
                        int seg_hi = seg_for_bead[mov_e];
                        int n_aff_full = seg_hi - seg_lo + 1;
                        int use_full_bias_recompute =
                            (n_aff_full > MAX_AFFECTED_SEGS) ? 1 : 0;

                        int s_end = seg_hi;
                        double old_coms[MAX_AFFECTED_SEGS][3];

                        if (!use_full_bias_recompute) {
                            s_end = seg_lo + n_aff_full - 1;
                            for (int s = seg_lo; s <= s_end; s++) {
                                old_coms[s - seg_lo][0] = segment_com[lms][s][0];
                                old_coms[s - seg_lo][1] = segment_com[lms][s][1];
                                old_coms[s - seg_lo][2] = segment_com[lms][s][2];
                            }
                        }

                        for (int i = mov_s; i <= mov_e; i++) {
                            double x = positions[lms][i][0] - ox;
                            double y = positions[lms][i][1] - oy;
                            double z = positions[lms][i][2] - oz;
                            positions[lms][i][0] = RM[0][0]*x + RM[0][1]*y + RM[0][2]*z + ox;
                            positions[lms][i][1] = RM[1][0]*x + RM[1][1]*y + RM[1][2]*z + oy;
                            positions[lms][i][2] = RM[2][0]*x + RM[2][1]*y + RM[2][2]*z + oz;
                        }

                        double new_bl = 0.0, new_br = 0.0;
                        {
                            double *pf = positions[lms][b1];
                            double *pm = positions[lms][mov_s];
                            double ddx = pf[0] - pm[0];
                            double ddy = pf[1] - pm[1];
                            double ddz = pf[2] - pm[2];
                            double dr  = sqrt(ddx*ddx + ddy*ddy + ddz*ddz) - bref[b1];
                            new_bl = kharmonic * dr * dr;
                        }
                        if (b2 < NUM_BEADS) {
                            double *pf = positions[lms][b2];
                            double *pm = positions[lms][mov_e];
                            double ddx = pm[0] - pf[0];
                            double ddy = pm[1] - pf[1];
                            double ddz = pm[2] - pf[2];
                            double dr  = sqrt(ddx*ddx + ddy*ddy + ddz*ddz) - bref[mov_e];
                            new_br = kharmonic * dr * dr;
                        }
                        double new_bond = saved_bond + (new_bl + new_br) - (old_bl + old_br);

                        double new_bend_bd = 0.0;
                        for (int k = 0; k < n_ba; k++) {
                            new_bend_bd += angle_energy_triplet(
                                positions[lms][P1[ba_list[k]]],
                                positions[lms][P2[ba_list[k]]],
                                positions[lms][P3[ba_list[k]]]
                            );
                        }
                        double new_bend = saved_bend + new_bend_bd - old_bend_bd;

                        for (int k = 0; k < n_moved; k++) {
                            bvh_query_sphere(wroot, widx, positions[lms],
                                             positions[lms][mov_s + k][0],
                                             positions[lms][mov_s + k][1],
                                             positions[lms][mov_s + k][2],
                                             MAX_RCUT, mov_s, mov_e,
                                             cands, &ncands, flags);
                        }

                        double old_lj_p = 0.0, new_lj_p = 0.0;
                        for (int ci = 0; ci < ncands; ci++) {
                            int j = cands[ci];
                            for (int k = 0; k < n_moved; k++) {
                                int i = mov_s + k;
                                if (pair_is_excluded_topologically(i, j)) continue;

                                {
                                    double dx = old_p[k][0] - positions[lms][j][0];
                                    double dy = old_p[k][1] - positions[lms][j][1];
                                    double dz = old_p[k][2] - positions[lms][j][2];
                                    double r2 = dx*dx + dy*dy + dz*dz;
                                    if (r2 < 1e-12) r2 = 1e-12;
                                    if (pair_can_interact(i, j, r2))
                                        old_lj_p += soft_repulsion_pair(i, j, r2);
                                }

                                {
                                    double dx = positions[lms][i][0] - positions[lms][j][0];
                                    double dy = positions[lms][i][1] - positions[lms][j][1];
                                    double dz = positions[lms][i][2] - positions[lms][j][2];
                                    double r2 = dx*dx + dy*dy + dz*dz;
                                    if (r2 < 1e-12) r2 = 1e-12;
                                    if (pair_can_interact(i, j, r2))
                                        new_lj_p += soft_repulsion_pair(i, j, r2);
                                }
                            }
                        }
                        double new_lj = saved_lj + (new_lj_p - old_lj_p);

                        double new_rb;

                        if (!use_full_bias_recompute) {
                            update_partial_segment_coms(ms, seg_lo, s_end);

                            double delta_rb = 0.0;

                            for (int s = seg_lo; s <= s_end; s++) {
                                for (int j2 = 0; j2 < NUM_SEGMENTS; j2++) {
                                    if (j2 >= seg_lo && j2 <= s_end) continue;
                                    int lo = (s < j2) ? s : j2;
                                    int hi = (s < j2) ? j2 : s;
                                    if (expC[lo][hi] <= 1e-10) continue;

                                    double lam = lamda[lo][hi];
                                    double ddx, ddy, ddz;

                                    ddx = old_coms[s - seg_lo][0] - segment_com[lms][j2][0];
                                    ddy = old_coms[s - seg_lo][1] - segment_com[lms][j2][1];
                                    ddz = old_coms[s - seg_lo][2] - segment_com[lms][j2][2];
                                    double w_old = contact_weight(sqrt(ddx*ddx + ddy*ddy + ddz*ddz));

                                    ddx = segment_com[lms][s][0] - segment_com[lms][j2][0];
                                    ddy = segment_com[lms][s][1] - segment_com[lms][j2][1];
                                    ddz = segment_com[lms][s][2] - segment_com[lms][j2][2];
                                    double w_new = contact_weight(sqrt(ddx*ddx + ddy*ddy + ddz*ddz));

                                    delta_rb += lam * (w_new - w_old);
                                }
                            }

                            for (int s1 = seg_lo; s1 <= s_end; s1++) {
                                for (int s2 = s1 + 1; s2 <= s_end; s2++) {
                                    if (expC[s1][s2] <= 1e-10) continue;

                                    double lam = lamda[s1][s2];
                                    double ddx, ddy, ddz;

                                    ddx = old_coms[s1 - seg_lo][0] - old_coms[s2 - seg_lo][0];
                                    ddy = old_coms[s1 - seg_lo][1] - old_coms[s2 - seg_lo][1];
                                    ddz = old_coms[s1 - seg_lo][2] - old_coms[s2 - seg_lo][2];
                                    double w_old = contact_weight(sqrt(ddx*ddx + ddy*ddy + ddz*ddz));

                                    ddx = segment_com[lms][s1][0] - segment_com[lms][s2][0];
                                    ddy = segment_com[lms][s1][1] - segment_com[lms][s2][1];
                                    ddz = segment_com[lms][s1][2] - segment_com[lms][s2][2];
                                    double w_new = contact_weight(sqrt(ddx*ddx + ddy*ddy + ddz*ddz));

                                    delta_rb += lam * (w_new - w_old);
                                }
                            }

                            new_rb = saved_rb + delta_rb;
                        } else {
                            update_all_segment_coms(ms);
                            new_rb = compute_raw_bias_from_coms(ms);
                        }

                        {
                            double new_bias = bias_beta * new_rb;
                            new_energy[lms] = new_bond + new_bend + new_lj + new_bias;

                            check_energy_finite(new_bond, "new_bond", move, ms);
                            check_energy_finite(new_bend, "new_bend", move, ms);
                            check_energy_finite(new_lj,   "new_lj",   move, ms);
                            check_energy_finite(new_rb,   "new_rb",   move, ms);
                            check_energy_finite(new_energy[lms], "new_energy", move, ms);

                            double dE = new_energy[lms] - old_energy[lms];
                            double r  = (double)rand_r(&seed_array[tid]) /
                                        ((double)RAND_MAX + 1.0);

                            if (dE <= 0.0 || r < exp(-dE / TEMPERATURE)) {
                                old_energy[lms]         = new_energy[lms];
                                bond_energy_walker[lms] = new_bond;
                                bend_energy_walker[lms] = new_bend;
                                lj_energy_walker[lms]   = new_lj;
                                raw_bias_walker[lms]    = new_rb;
                                check_positions_finite(ms, "accepted crankshaft move", move);
                                rebuild_walker_bvh(ms);
                                accepted_count_local++;
                            } else {
                                for (int k = 0; k < n_moved; k++) {
                                    int bk = mov_s + k;
                                    positions[lms][bk][0] = old_p[k][0];
                                    positions[lms][bk][1] = old_p[k][1];
                                    positions[lms][bk][2] = old_p[k][2];
                                }

                                if (!use_full_bias_recompute) {
                                    for (int s = seg_lo; s <= s_end; s++) {
                                        segment_com[lms][s][0] = old_coms[s - seg_lo][0];
                                        segment_com[lms][s][1] = old_coms[s - seg_lo][1];
                                        segment_com[lms][s][2] = old_coms[s - seg_lo][2];
                                    }
                                } else {
                                    update_all_segment_coms(ms);
                                }
                            }
                        }
                    }

                } else {
                    /* ========================================================
 *                      * PIVOT / GAUSSIAN branch
 *                                           * ======================================================== */

                    for (int bi = 0; bi < NUM_BEADS; bi++) {
                        old_positions[lms][bi][0] = positions[lms][bi][0];
                        old_positions[lms][bi][1] = positions[lms][bi][1];
                        old_positions[lms][bi][2] = positions[lms][bi][2];
                    }

                    if (ran < probA)
                        pivot_move(tid, lms, positions, seed_array);
                    else
                        gaussian_displacement(tid, lms, positions, seed_array);

                    update_all_segment_coms(ms);

                    {
                        double new_bond = calculate_bond_energy(positions, lms);
                        double new_bend = calculate_bend_energy(positions, lms);
                        double new_lj   = compute_lj_energy_bvh(positions[lms], NUM_BEADS, tid);
                        double new_rb   = compute_raw_bias_from_coms(ms);

                        new_energy[lms] = new_bond + new_bend + new_lj + bias_beta * new_rb;

                        check_energy_finite(new_bond, "new_bond", move, ms);
                        check_energy_finite(new_bend, "new_bend", move, ms);
                        check_energy_finite(new_lj,   "new_lj",   move, ms);
                        check_energy_finite(new_rb,   "new_rb",   move, ms);
                        check_energy_finite(new_energy[lms], "new_energy", move, ms);

                        double dE = new_energy[lms] - old_energy[lms];
                        double r  = (double)rand_r(&seed_array[tid]) /
                                    ((double)RAND_MAX + 1.0);

                        if (dE <= 0.0 || r < exp(-dE / TEMPERATURE)) {
                            old_energy[lms]         = new_energy[lms];
                            bond_energy_walker[lms] = new_bond;
                            bend_energy_walker[lms] = new_bend;
                            lj_energy_walker[lms]   = new_lj;
                            raw_bias_walker[lms]    = new_rb;
                            check_positions_finite(ms, "accepted move", move);
                            rebuild_walker_bvh(ms);
                            accepted_count_local++;
                        } else {
                            for (int bi = 0; bi < NUM_BEADS; bi++) {
                                positions[lms][bi][0] = old_positions[lms][bi][0];
                                positions[lms][bi][1] = old_positions[lms][bi][1];
                                positions[lms][bi][2] = old_positions[lms][bi][2];
                            }
                            update_all_segment_coms(ms);
                        }
                    }
                }

NEXT_WALKER:
                ;
            } /* end walker loop */

#pragma omp single
            {
                int sip = move - phase_start_step - 1;

                if (bias_phase_state == 0) {
                    bias_beta *= 1.005;
                    if (bias_beta >= bb_MAX) bias_beta = bb_MAX;

                    rebase_total_energy_from_components(ms_start, ms_end);

                    if (move + 1 == RB_STEP) {
                        bias_phase_state = 1;
                        phase_start_step = move;
                    }

                } else if (bias_phase_state == 1) {
                    if (sip == 0) {
                        if (mpi_rank == 0)
                            printf("*** Debiasing started (move %d) ***\n", move);

                        bias_beta   = 0.001;
                        TEMPERATURE = 2.0;

                        rebase_total_energy_from_components(ms_start, ms_end);
                    }

                    if (sip + 1 >= DB_STEP) {
                        bias_phase_state = 2;
                        phase_start_step = move;
                    }

                } else { /* rebiasing */
                    if (sip == 0) {
                        if (mpi_rank == 0)
                            printf("*** Rebiasing started (move %d) ***\n", move);

                        TEMPERATURE = 1.0;
                    }

                    if (sip < StepslowBBupdate)
                        bias_beta *= 1.01;
                    else
                        bias_beta *= 1.05;

                    if (bias_beta > bb_MAX) bias_beta = bb_MAX;

                    rebase_total_energy_from_components(ms_start, ms_end);

                    if (sip + 1 >= RB_STEP) {
                        bias_phase_state = 1;
                        phase_start_step = move;
                    }
                }
            } /* end omp single */

        } /* end omp parallel */

        /* ── global acceptance report ───────────────────────── */
        {
            int accepted_global = 0;
            MPI_Allreduce(&accepted_count_local, &accepted_global,
                          1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

            if (mpi_rank == 0) {
                printf("move %d | phase=%d | bb=%.5f | T=%.3f | acc=%.4f\n",
                       move, bias_phase_state, bias_beta, TEMPERATURE,
                       (double)accepted_global / (double)Tsnap);
            }
        }

        /* ── iterative MaxEnt update every 100 moves using all 6000 conformations ── */
        if ((move + 1) % LAMBDA_UPDATE_INTERVAL == 0) {

            calculate_contact_map_mpi(ms_start, ms_end, 1);
            if ((move + 201) % 1400 == 0) {

            if (mpi_rank == 0) {
                static int iter_id = 0;
                char cmapfile[256], lamfile[256];
                FILE *fcm, *flam;

                iter_id++;

                sprintf(cmapfile, "Output/simC_iter_%04d.txt", iter_id);
                sprintf(lamfile,  "Output/lambda_iter_%04d.txt", iter_id);

                fcm = fopen(cmapfile, "w");
                if (fcm) {
                    for (int ii = 0; ii < NUM_SEGMENTS; ii++) {
                        for (int jj = 0; jj < NUM_SEGMENTS; jj++) {
                            fprintf(fcm, "%d %d %.10f\n", ii, jj,
                                    (ii == jj) ? 0.0 : simC[ii][jj]);
                        }
                        fprintf(fcm, "\n");
                    }
                    fclose(fcm);
                }

                flam = fopen(lamfile, "w");
                if (flam) {
                    for (int ii = 0; ii < NUM_SEGMENTS; ii++) {
                        for (int jj = 0; jj < NUM_SEGMENTS; jj++) {
                            fprintf(flam, "%.10f\n", lamda[ii][jj]);
                        }
                    }
                    fclose(flam);
                }

                printf("Iterative MaxEnt update written: %s , %s\n",
                       cmapfile, lamfile);
            }
	  }
        }

    } /* end move loop */

    if (mpi_rank == 0)
        printf("Run complete. Writing final outputs...\n");

    /* final contact map without further lambda update */
    calculate_contact_map_mpi(ms_start, ms_end, 0);

    if (mpi_rank == 0) {
        FILE *fcm = fopen("Output/final_contact_map.txt", "w");
        if (fcm) {
            for (int ii = 0; ii < NUM_SEGMENTS; ii++) {
                for (int jj = 0; jj < NUM_SEGMENTS; jj++) {
                    fprintf(fcm, "%d %d %.10f\n", ii, jj,
                            (ii == jj) ? 0.0 : simC[ii][jj]);
                }
                fprintf(fcm, "\n");
            }
            fclose(fcm);
            printf("Final contact map written.\n");
        }

        FILE *flam = fopen("Output/final_lambda_matrix.txt", "w");
        if (flam) {
            for (int ii = 0; ii < NUM_SEGMENTS; ii++) {
                for (int jj = 0; jj < NUM_SEGMENTS; jj++) {
                    fprintf(flam, "%.10f\n", lamda[ii][jj]);
                }
            }
            fclose(flam);
            printf("Final lambda matrix written.\n");
        }
    }

    /* final per-rank snapshot */
    {
        char fname[256];
        FILE *snap;

        sprintf(fname, "%s_rank%03d.xyz", WRITEPOS, mpi_rank);
        snap = fopen(fname, "w");
        if (snap) {
            fprintf(snap, "# Final configurations rank %d\n", mpi_rank);
            for (int ms = ms_start; ms < ms_end; ms++) {
                int lms = LIDX(ms);
                for (int b = 0; b < NUM_BEADS; b++) {
                    fprintf(snap, "%.10f %.10f %.10f\n",
                            positions[lms][b][0],
                            positions[lms][b][1],
                            positions[lms][b][2]);
                }
            }
            fclose(snap);
        }
    }

    free(seed_array);
    cleanup();

    MPI_Finalize();
    return 0;
}
