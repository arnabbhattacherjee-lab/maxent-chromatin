#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include "sys_c.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef MAX_BEADS_IN_LEAF
#define MAX_BEADS_IN_LEAF 8
#endif

#ifndef MAX_RADIUS
#define MAX_RADIUS 5.0
#endif

#define MAX_SIGMA (2.0 * MAX_RADIUS)
//##define MAX_RCUT  (2.5 * MAX_SIGMA)

/* ---------------- globals ---------------- */

double cut = 30.0;
double r_cutoff = 15.0;
double epsilon = 1.0;
double limit = 1.0;
double limit2 = 400.0;
double bias_strength = 0.3;
double prob_cut = 0.85;
int del;

double *rad, *bref;
double **positions, **old_positions, **contact_freq;
int *btyp;
int N_angle = 0;
int *P1, *P2, *P3;

/* ---------------- prototypes ---------------- */

double calculate_bond_energy(void);
double calculate_bend_energy(void);
double compute_lj_energy_bvh(double **pos, int num_beads);

void gaussian_displacement(void);
void pivot_move(void);
void crankshaft_move(void);
void rigid_body_rotation(void);
void get_the_old_position(void);
void write_positions_to_file(const char *filename, const char *typ);
void runtime(long it, double o[], int n);

double rand_gaussian(void);

/* ---------------- BVH ---------------- */

typedef struct BVHNode {
    double bbox_min[3], bbox_max[3];
    int start, count;
    struct BVHNode *left, *right;
    int is_leaf;
} BVHNode;

typedef struct {
    double **positions;
    int axis;
} SortContext;

static SortContext g_sort_ctx;

/* ---------------- utilities ---------------- */

static inline int pair_is_excluded_topologically(int i, int j)
{
    int d = abs(i - j);
    if (d <= 1) return 1;
    if (d == 2 && btyp[i] == 0 && btyp[j] == 0) return 1;
    return 0;
}

static inline double pair_sigma(int i, int j)
{
    return rad[i] + rad[j];
}

static inline double pair_rcut_from_sigma(double sigma)
{
    return (fabs(sigma - 10.0) < 1e-12) ? (2.5 * sigma) : (1.1225 * sigma);
}

static inline double pair_rcut(int i, int j)
{
    return pair_rcut_from_sigma(pair_sigma(i, j));
}

static inline int pair_can_interact(int i, int j, double r2)
{
    double rc = pair_rcut(i, j);
    return (r2 < rc * rc);
}

static inline double soft_repulsion_pair(int i, int j, double r2)
{
    double sigma = pair_sigma(i, j);
    double rcut  = pair_rcut_from_sigma(sigma);
    double sig2  = sigma * sigma;
    double rcut2 = rcut * rcut;

    if (r2 >= rcut2) return 0.0;
    if (r2 < 1e-12) r2 = 1e-12;

    double r2ic  = sig2 / rcut2;
    double r6ic  = r2ic * r2ic * r2ic;
    double r12ic = r6ic * r6ic;

    double r2i   = sig2 / r2;
    double r6i   = r2i * r2i * r2i;
    double r12i  = r6i * r6i;

    return 0.1 * ((r12i - r6i) - (r12ic - r6ic));
}

double rand_gaussian(void)
{
    double u = (double)rand() / (double)RAND_MAX;
    double v = (double)rand() / (double)RAND_MAX;
    if (u < 1e-12) u = 1e-12;
    return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
}

static int compare_beads(const void *a, const void *b)
{
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    double va = g_sort_ctx.positions[ia][g_sort_ctx.axis];
    double vb = g_sort_ctx.positions[ib][g_sort_ctx.axis];
    return (va < vb) ? -1 : (va > vb) ? 1 : 0;
}

void compute_bounding_box(int *indices, int start, int count,
                          double *bbox_min, double *bbox_max, double **pos)
{
    for (int d = 0; d < 3; ++d) {
        bbox_min[d] = DBL_MAX;
        bbox_max[d] = -DBL_MAX;
    }

    for (int i = start; i < start + count; ++i) {
        int idx = indices[i];
        for (int d = 0; d < 3; ++d) {
            double p = pos[idx][d];
            double r = rad[idx];
            if (p - r < bbox_min[d]) bbox_min[d] = p - r;
            if (p + r > bbox_max[d]) bbox_max[d] = p + r;
        }
    }
}

static BVHNode *build_bvh(int *indices, int start, int count, double **pos)
{
    BVHNode *node = (BVHNode *)malloc(sizeof(BVHNode));
    if (!node) {
        fprintf(stderr, "BVH allocation failed\n");
        exit(EXIT_FAILURE);
    }

    node->start = start;
    node->count = count;
    node->left = NULL;
    node->right = NULL;

    compute_bounding_box(indices, start, count, node->bbox_min, node->bbox_max, pos);

    if (count <= MAX_BEADS_IN_LEAF) {
        node->is_leaf = 1;
        return node;
    }

    node->is_leaf = 0;

    int axis = 0;
    double max_len = node->bbox_max[0] - node->bbox_min[0];
    for (int d = 1; d < 3; ++d) {
        double len = node->bbox_max[d] - node->bbox_min[d];
        if (len > max_len) {
            max_len = len;
            axis = d;
        }
    }

    g_sort_ctx.positions = pos;
    g_sort_ctx.axis = axis;
    qsort(indices + start, (size_t)count, sizeof(int), compare_beads);

    int left_count = count / 2;
    int right_count = count - left_count;
    int mid = start + left_count;

    node->left  = build_bvh(indices, start, left_count, pos);
    node->right = build_bvh(indices, mid, right_count, pos);

    return node;
}

static inline int boxes_overlap(const double *min1, const double *max1,
                                const double *min2, const double *max2,
                                double margin)
{
    for (int d = 0; d < 3; ++d) {
        if (min1[d] > max2[d] + margin) return 0;
        if (max1[d] < min2[d] - margin) return 0;
    }
    return 1;
}

static void process_leaf_self(BVHNode *leaf, double *energy, double **pos, int *indices)
{
    int s = leaf->start;
    int e = leaf->start + leaf->count;

    for (int a = s; a < e; ++a) {
        int i = indices[a];
        for (int b = a + 1; b < e; ++b) {
            int j = indices[b];

            if (pair_is_excluded_topologically(i, j)) continue;

            double dx = pos[i][0] - pos[j][0];
            double dy = pos[i][1] - pos[j][1];
            double dz = pos[i][2] - pos[j][2];
            double r2 = dx*dx + dy*dy + dz*dz;

            if (pair_can_interact(i, j, r2))
                *energy += soft_repulsion_pair(i, j, r2);
        }
    }
}

static void process_leaf_cross(BVHNode *a, BVHNode *b, double *energy,
                               double **pos, int *indices)
{
    int as = a->start, ae = a->start + a->count;
    int bs = b->start, be = b->start + b->count;

    for (int ia = as; ia < ae; ++ia) {
        int i = indices[ia];
        for (int ib = bs; ib < be; ++ib) {
            int j = indices[ib];

            if (pair_is_excluded_topologically(i, j)) continue;

            double dx = pos[i][0] - pos[j][0];
            double dy = pos[i][1] - pos[j][1];
            double dz = pos[i][2] - pos[j][2];
            double r2 = dx*dx + dy*dy + dz*dz;

            if (pair_can_interact(i, j, r2))
                *energy += soft_repulsion_pair(i, j, r2);
        }
    }
}

static void traverse_bvh_cross(BVHNode *a, BVHNode *b, double *energy,
                               double **pos, int *indices)
{
    if (!boxes_overlap(a->bbox_min, a->bbox_max, b->bbox_min, b->bbox_max, MAX_RCUT))
        return;

    if (a->is_leaf && b->is_leaf) {
        process_leaf_cross(a, b, energy, pos, indices);
        return;
    }

    if (a->is_leaf) {
        traverse_bvh_cross(a, b->left,  energy, pos, indices);
        traverse_bvh_cross(a, b->right, energy, pos, indices);
    } else if (b->is_leaf) {
        traverse_bvh_cross(a->left,  b, energy, pos, indices);
        traverse_bvh_cross(a->right, b, energy, pos, indices);
    } else if (a->count >= b->count) {
        traverse_bvh_cross(a->left,  b, energy, pos, indices);
        traverse_bvh_cross(a->right, b, energy, pos, indices);
    } else {
        traverse_bvh_cross(a, b->left,  energy, pos, indices);
        traverse_bvh_cross(a, b->right, energy, pos, indices);
    }
}

static void traverse_bvh_self(BVHNode *node, double *energy, double **pos, int *indices)
{
    if (node->is_leaf) {
        process_leaf_self(node, energy, pos, indices);
        return;
    }

    traverse_bvh_self(node->left, energy, pos, indices);
    traverse_bvh_cross(node->left, node->right, energy, pos, indices);
    traverse_bvh_self(node->right, energy, pos, indices);
}

static void free_bvh(BVHNode *node)
{
    if (!node) return;
    free_bvh(node->left);
    free_bvh(node->right);
    free(node);
}

double compute_lj_energy_bvh(double **pos, int num_beads)
{
    int *indices = (int *)malloc((size_t)num_beads * sizeof(int));
    if (!indices) {
        fprintf(stderr, "indices allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_beads; ++i) indices[i] = i;

    BVHNode *root = build_bvh(indices, 0, num_beads, pos);

    double energy = 0.0;
    traverse_bvh_self(root, &energy, pos, indices);

    free_bvh(root);
    free(indices);
    return energy;
}

/* ---------------- MC move helpers ---------------- */

void get_the_old_position(void)
{
    for (int i = 0; i < NUM_BEADS; i++) {
        positions[i][0] = old_positions[i][0];
        positions[i][1] = old_positions[i][1];
        positions[i][2] = old_positions[i][2];
    }
}

static void save_old_positions(void)
{
    for (int i = 0; i < NUM_BEADS; i++) {
        old_positions[i][0] = positions[i][0];
        old_positions[i][1] = positions[i][1];
        old_positions[i][2] = positions[i][2];
    }
}

void gaussian_displacement(void)
{
    double sigma = 0.3;
    int max_length = 1000;

    save_old_positions();

    int bead_index1 = rand() % NUM_BEADS;
    int length = 1 + rand() % max_length;
    int bead_index2 = bead_index1 + length;
    if (bead_index2 >= NUM_BEADS) bead_index2 = NUM_BEADS - 1;

    int start = (bead_index1 > 0) ? bead_index1 : 1;
    int end   = (bead_index2 < NUM_BEADS - 1) ? (bead_index2 + 1) : (NUM_BEADS - 1);

    for (int i = start; i <= end; i++) {
        positions[i][0] += sigma * rand_gaussian();
        positions[i][1] += sigma * rand_gaussian();
        positions[i][2] += sigma * rand_gaussian();
    }

    for (int i = start; i < NUM_BEADS; i++) {
        double dx = positions[i][0] - positions[i - 1][0];
        double dy = positions[i][1] - positions[i - 1][1];
        double dz = positions[i][2] - positions[i - 1][2];
        double dist = sqrt(dx*dx + dy*dy + dz*dz);
        if (dist < 1e-12) dist = 1e-12;

        double scale = bref[i - 1] / dist;
        positions[i][0] = positions[i - 1][0] + dx * scale;
        positions[i][1] = positions[i - 1][1] + dy * scale;
        positions[i][2] = positions[i - 1][2] + dz * scale;
    }
}

void pivot_move(void)
{
    save_old_positions();

    int pivot_point = rand() % NUM_BEADS;
    if (pivot_point == NUM_BEADS - 1) return;

    int rotate_left = (rand() & 1);
    int S, E;

    if (rotate_left) {
        if (pivot_point <= 0) return;
        S = 0;
        E = pivot_point;
    } else {
        if (pivot_point >= NUM_BEADS - 2) return;
        S = pivot_point + 1;
        E = NUM_BEADS;
    }

    double ux = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
    double uy = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
    double uz = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
    double norm = sqrt(ux*ux + uy*uy + uz*uz);
    if (norm < 1e-12) return;
    ux /= norm; uy /= norm; uz /= norm;

    double angle = 2.0 * M_PI * ((double)rand() / RAND_MAX);
    double c = cos(angle), s = sin(angle), omc = 1.0 - c;

    double R[3][3];
    R[0][0] = c + ux*ux*omc;
    R[0][1] = ux*uy*omc - uz*s;
    R[0][2] = ux*uz*omc + uy*s;
    R[1][0] = uy*ux*omc + uz*s;
    R[1][1] = c + uy*uy*omc;
    R[1][2] = uy*uz*omc - ux*s;
    R[2][0] = uz*ux*omc - uy*s;
    R[2][1] = uz*uy*omc + ux*s;
    R[2][2] = c + uz*uz*omc;

    double ox = positions[pivot_point][0];
    double oy = positions[pivot_point][1];
    double oz = positions[pivot_point][2];

    for (int i = S; i < E; i++) {
        double x = positions[i][0] - ox;
        double y = positions[i][1] - oy;
        double z = positions[i][2] - oz;

        double nxp = R[0][0]*x + R[0][1]*y + R[0][2]*z;
        double nyp = R[1][0]*x + R[1][1]*y + R[1][2]*z;
        double nzp = R[2][0]*x + R[2][1]*y + R[2][2]*z;

        positions[i][0] = nxp + ox;
        positions[i][1] = nyp + oy;
        positions[i][2] = nzp + oz;
    }
}

void crankshaft_move(void)
{
    save_old_positions();

    int max_length = 20;
    int bead1 = rand() % NUM_BEADS;
    int length = 2 + rand() % (max_length - 1);
    int bead2 = bead1 + length;

    if (bead2 >= NUM_BEADS) bead2 = NUM_BEADS - 1;
    if (bead2 - bead1 < 2) return;

    double dx = positions[bead2][0] - positions[bead1][0];
    double dy = positions[bead2][1] - positions[bead1][1];
    double dz = positions[bead2][2] - positions[bead1][2];
    double r  = sqrt(dx*dx + dy*dy + dz*dz);
    if (r < 1e-12) return;

    double nx = dx / r, ny = dy / r, nz = dz / r;
    double angle = 2.0 * M_PI * ((double)rand() / RAND_MAX);
    double c = cos(angle), s = sin(angle), omc = 1.0 - c;

    double R[3][3];
    R[0][0] = c + nx*nx*omc;
    R[0][1] = nx*ny*omc - nz*s;
    R[0][2] = nx*nz*omc + ny*s;
    R[1][0] = ny*nx*omc + nz*s;
    R[1][1] = c + ny*ny*omc;
    R[1][2] = ny*nz*omc - nx*s;
    R[2][0] = nz*nx*omc - ny*s;
    R[2][1] = nz*ny*omc + nx*s;
    R[2][2] = c + nz*nz*omc;

    double ox = positions[bead1][0];
    double oy = positions[bead1][1];
    double oz = positions[bead1][2];

    for (int i = bead1 + 1; i < bead2; i++) {
        double x = positions[i][0] - ox;
        double y = positions[i][1] - oy;
        double z = positions[i][2] - oz;

        double nxp = R[0][0]*x + R[0][1]*y + R[0][2]*z;
        double nyp = R[1][0]*x + R[1][1]*y + R[1][2]*z;
        double nzp = R[2][0]*x + R[2][1]*y + R[2][2]*z;

        positions[i][0] = nxp + ox;
        positions[i][1] = nyp + oy;
        positions[i][2] = nzp + oz;
    }
}

void rigid_body_rotation(void)
{
    save_old_positions();

    double com[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < NUM_BEADS; i++) {
        com[0] += positions[i][0];
        com[1] += positions[i][1];
        com[2] += positions[i][2];
    }
    com[0] /= NUM_BEADS;
    com[1] /= NUM_BEADS;
    com[2] /= NUM_BEADS;

    double axis[3] = {
        rand() / (double)RAND_MAX,
        rand() / (double)RAND_MAX,
        rand() / (double)RAND_MAX
    };

    double norm = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    if (norm < 1e-12) return;

    axis[0] /= norm;
    axis[1] /= norm;
    axis[2] /= norm;

    double angle = 0.1;
    double c = cos(angle);
    double s = sin(angle);
    double omc = 1.0 - c;

    for (int i = 0; i < NUM_BEADS; i++) {
        double x = positions[i][0] - com[0];
        double y = positions[i][1] - com[1];
        double z = positions[i][2] - com[2];

        double dot = axis[0]*x + axis[1]*y + axis[2]*z;
        double cross[3] = {
            axis[1]*z - axis[2]*y,
            axis[2]*x - axis[0]*z,
            axis[0]*y - axis[1]*x
        };

        double rx = x*c + cross[0]*s + axis[0]*dot*omc;
        double ry = y*c + cross[1]*s + axis[1]*dot*omc;
        double rz = z*c + cross[2]*s + axis[2]*dot*omc;

        positions[i][0] = com[0] + rx;
        positions[i][1] = com[1] + ry;
        positions[i][2] = com[2] + rz;
    }
}

/* ---------------- energies ---------------- */

double calculate_bond_energy(void)
{
    double bond_energy = 0.0;
    double kh = 0.1;

    for (int i = 0; i < NUM_BEADS - 1; i++) {
        double dx = positions[i + 1][0] - positions[i][0];
        double dy = positions[i + 1][1] - positions[i][1];
        double dz = positions[i + 1][2] - positions[i][2];
        double r = sqrt(dx*dx + dy*dy + dz*dz);
        double dr = r - bref[i];
        bond_energy += kh * dr * dr;
    }
    return bond_energy;
}

double calculate_bend_energy(void)
{
    double bend_energy = 0.0;
    double kb = 20.0;

    for (int i = 0; i < N_angle; i++) {
        int i1 = P1[i];
        int i2 = P2[i];
        int i3 = P3[i];

        double v1[3], v2[3];
        for (int d = 0; d < 3; d++) {
            v1[d] = positions[i2][d] - positions[i1][d];
            v2[d] = positions[i3][d] - positions[i2][d];
        }

        double dot  = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
        double mag1 = sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
        double mag2 = sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2]);

        if (mag1 > 1e-8 && mag2 > 1e-8) {
            double cos_theta = dot / (mag1 * mag2);
            if (cos_theta > 1.0) cos_theta = 1.0;
            if (cos_theta < -1.0) cos_theta = -1.0;
            bend_energy += (1.0 - cos_theta);
        }
    }

    return kb * bend_energy;
}

/* ---------------- output ---------------- */

void write_positions_to_file(const char *filename, const char *typ)
{
    FILE *file = fopen(filename, typ);
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        return;
    }

    for (int i = 0; i < NUM_BEADS; i++) {
        fprintf(file, "%f %f %f\n", positions[i][0], positions[i][1], positions[i][2]);
    }
    fclose(file);
}

void runtime(long it, double o[], int n)
{
    FILE *fp = fopen("tmp_rt", "w");
    if (!fp) return;

    fprintf(fp, "%li %f ", it, TEMPERATURE);
    for (int i = 1; i < n; i++) {
        if (i != 7) fprintf(fp, "%f ", o[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
}

/* ---------------- main ---------------- */

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;

    double old_energy, new_energy, BE, EV, AE;
    int move_accept_count = 0;
    double o[NOBS];

    rad          = (double *)calloc(NUM_BEADS, sizeof(double));
    bref         = (double *)calloc(NUM_BEADS, sizeof(double));
    btyp         = (int *)calloc(NUM_BEADS, sizeof(int));
    positions    = (double **)calloc(NUM_BEADS, sizeof(double *));
    old_positions= (double **)calloc(NUM_BEADS, sizeof(double *));
    contact_freq = (double **)calloc(NUM_BEADS, sizeof(double *));
    P1           = (int *)calloc(NUM_BEADS, sizeof(int));
    P2           = (int *)calloc(NUM_BEADS, sizeof(int));
    P3           = (int *)calloc(NUM_BEADS, sizeof(int));

    if (!rad || !bref || !btyp || !positions || !old_positions ||
        !contact_freq || !P1 || !P2 || !P3) {
        fprintf(stderr, "Allocation failed\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < NUM_BEADS; i++) {
        positions[i]     = (double *)calloc(3, sizeof(double));
        old_positions[i] = (double *)calloc(3, sizeof(double));
        contact_freq[i]  = (double *)calloc(NUM_BEADS, sizeof(double));

        if (!positions[i] || !old_positions[i] || !contact_freq[i]) {
            fprintf(stderr, "Per-bead allocation failed\n");
            return EXIT_FAILURE;
        }
    }

    FILE *nucpos = fopen(NUCPOS, "r");
    if (!nucpos) {
        fprintf(stderr, "Cannot open %s\n", NUCPOS);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < NUM_BEADS; i++) {
        if (fscanf(nucpos, "%d", &btyp[i]) != 1) {
            fprintf(stderr, "Error reading NUCPOS at bead %d\n", i);
            fclose(nucpos);
            return EXIT_FAILURE;
        }
        rad[i] = (btyp[i] == 0) ? 1.25 : 5.0;
    }
    fclose(nucpos);

    for (int i = 0; i < NUM_BEADS - 1; i++)
        bref[i] = rad[i] + rad[i+1];

    FILE *file3 = fopen(ANGINFO, "w");
    if (!file3) {
        fprintf(stderr, "Cannot open %s for writing\n", ANGINFO);
        return EXIT_FAILURE;
    }

    N_angle = 0;
    for (int i = 0; i < NUM_BEADS - 2; i++) {
        if (btyp[i] == 0 && btyp[i+1] == 0 && btyp[i+2] == 0) {
            P1[N_angle] = i;
            P2[N_angle] = i + 1;
            P3[N_angle] = i + 2;
            fprintf(file3, "%d\t%d\t%d\n", P1[N_angle], P2[N_angle], P3[N_angle]);
            N_angle++;
        }
    }
    fclose(file3);

    positions[0][0] = 0.0;
    positions[0][1] = 0.0;
    positions[0][2] = 0.0;

    {
        double FLUCTUATION = 0.5;
        for (int i = 1; i < NUM_BEADS; i++) {
            double bond = bref[i - 1];
            double fluct_y = (((double)rand() / (double)RAND_MAX) * 2.0 - 1.0) * FLUCTUATION * bond;
            double fluct_z = (((double)rand() / (double)RAND_MAX) * 2.0 - 1.0) * FLUCTUATION * bond;
            double x2 = bond*bond - fluct_y*fluct_y - fluct_z*fluct_z;
            if (x2 < 1e-12) x2 = 1e-12;

            positions[i][0] = positions[i-1][0] + sqrt(x2);
            positions[i][1] = positions[i-1][1] + fluct_y;
            positions[i][2] = positions[i-1][2] + fluct_z;
        }
    }

    FILE *file1 = fopen(STRTPOS, "w");
    if (file1) {
        for (int i = 0; i < NUM_BEADS; i++) {
            fprintf(file1, "%f %f %f\n", positions[i][0], positions[i][1], positions[i][2]);
        }
        fclose(file1);
    }

    srand((unsigned int)time(NULL));

    BE = calculate_bond_energy();
    EV = compute_lj_energy_bvh(positions, NUM_BEADS);
    AE = calculate_bend_energy();

    old_energy = BE + EV + AE;
    new_energy = old_energy;

    printf("----> Ebond=%f \t Ebend=%f \t Eev=%f \t Etotal=%f\n", BE, AE, EV, old_energy);

    for (int move = 0; move < MAX_MOVES; move++) {
        for (int ms = 0; ms < MOVE_steps; ms++) {
            double move_type = (double)rand() / ((double)RAND_MAX + 1.0);

            if (move_type < 0.33) {
                gaussian_displacement();
            } else if (move_type < 0.67) {
                pivot_move();
            } else {
                crankshaft_move();
            }

            BE = calculate_bond_energy();
            EV = compute_lj_energy_bvh(positions, NUM_BEADS);
            AE = calculate_bend_energy();
            new_energy = BE + EV + AE;

            if (new_energy <= old_energy ||
                exp((old_energy - new_energy) / TEMPERATURE) > ((double)rand() / RAND_MAX)) {
                move_accept_count++;
                old_energy = new_energy;
            } else {
                get_the_old_position();
                new_energy = old_energy;
            }

            o[1] = new_energy;
            o[2] = BE;
            o[3] = AE;
            o[4] = EV;
        }

        runtime(move, o, NOBS);

        if (move >= NTHERM && move % NFREQ == 0) {
            write_positions_to_file(WRITEPOS, "a");
        }

        printf("Moves accepted: %f\n",
               (double)move_accept_count / (double)((move + 1) * MOVE_steps));
    }

    printf("Simulation complete.\n");

    for (int i = 0; i < NUM_BEADS; i++) {
        free(positions[i]);
        free(old_positions[i]);
        free(contact_freq[i]);
    }

    free(positions);
    free(old_positions);
    free(contact_freq);
    free(rad);
    free(bref);
    free(btyp);
    free(P1);
    free(P2);
    free(P3);

    return 0;
}
