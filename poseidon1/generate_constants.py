#!/usr/bin/env python3
"""
Poseidon Round Constant Generator

Generates round constants, MDS matrix, and test vectors for the Poseidon hash
function (original version, a.k.a. Poseidon1) over various prime fields.

Reference: Grassi et al., "POSEIDON: A New Hash Function for Zero-Knowledge
Proof Systems" (https://eprint.iacr.org/2019/458)

Round structure (for width t, R_F full rounds, R_P partial rounds):

  Input → [R_F/2 full rounds] → [R_P partial rounds] → [R_F/2 full rounds] → Output

  Full round:    AddRoundConstants(all t) → S-box(all t) → MDS
  Partial round: AddRoundConstants(all t) → S-box(s0 only) → MDS

Unlike Poseidon2, every round (including partial rounds) uses t round constants,
and there is a single MDS matrix used in every round.

Supported fields:
  Field       Prime                  Bits  Alpha  Widths
  BabyBear    2013265921             31    7      16, 24
  KoalaBear   2130706433             31    3      16, 24
  Goldilocks  2^64 - 2^32 + 1       64    7      8, 12

Usage:
  python poseidon/generate_constants.py --field babybear --width 16
  python poseidon/generate_constants.py --field goldilocks --width 8 --format json
  python poseidon/generate_constants.py --field koalabear --width 24 -v
"""

import argparse
import json
import re
import sys
from math import ceil, floor, gcd, log, log2, inf
from pathlib import Path

# =============================================================================
# Field Definitions
# =============================================================================

FIELDS = {
    "babybear": {
        "prime": 2013265921,
        "valid_widths": [16, 24],
    },
    "koalabear": {
        "prime": 2130706433,
        "valid_widths": [16, 24],
    },
    "goldilocks": {
        "prime": (1 << 64) - (1 << 32) + 1,
        "valid_widths": [8, 12],
    },
    "mersenne31": {
        "prime": (1 << 31) - 1,
        "valid_widths": [16, 32],
    },
}

# Round-constant locations for in-tree reproducibility checks.
RUST_ROUND_CONSTANTS = {
    ("babybear", 16): ("baby-bear/src/poseidon1.rs", "BABYBEAR_POSEIDON1_RC_16"),
    ("babybear", 24): ("baby-bear/src/poseidon1.rs", "BABYBEAR_POSEIDON1_RC_24"),
    ("koalabear", 16): ("koala-bear/src/poseidon1.rs", "KOALABEAR_POSEIDON1_RC_16"),
    ("koalabear", 24): ("koala-bear/src/poseidon1.rs", "KOALABEAR_POSEIDON1_RC_24"),
    ("goldilocks", 8): ("goldilocks/src/poseidon1.rs", "GOLDILOCKS_POSEIDON1_RC_8"),
    ("goldilocks", 12): ("goldilocks/src/poseidon1.rs", "GOLDILOCKS_POSEIDON1_RC_12"),
}


def compute_alpha(p):
    """Smallest integer alpha >= 3 such that gcd(alpha, p-1) = 1."""
    for alpha in range(3, p):
        if gcd(alpha, p - 1) == 1:
            return alpha
    raise ValueError(f"No valid alpha found for p={p}")


# =============================================================================
# Finite Field Arithmetic
# =============================================================================


def extended_gcd(a, b):
    """Extended Euclidean algorithm. Returns (gcd, x, y) where a*x + b*y = gcd."""
    if a == 0:
        return b, 0, 1
    g, x, y = extended_gcd(b % a, a)
    return g, y - (b // a) * x, x


def mod_inv(a, p):
    """Modular inverse of a mod p via extended Euclidean algorithm."""
    if a == 0:
        raise ZeroDivisionError("Cannot invert zero")
    g, x, _ = extended_gcd(a % p, p)
    if g != 1:
        raise ValueError(f"{a} has no inverse mod {p}")
    return x % p


# =============================================================================
# Matrix Operations over GF(p)
# =============================================================================


def mat_identity(t, p):
    """Create t x t identity matrix over GF(p)."""
    M = [[0] * t for _ in range(t)]
    for i in range(t):
        M[i][i] = 1
    return M


def mat_mul(A, B, p):
    """Matrix multiplication mod p."""
    t = len(A)
    C = [[0] * t for _ in range(t)]
    for i in range(t):
        for j in range(t):
            s = 0
            for k in range(t):
                s += A[i][k] * B[k][j]
            C[i][j] = s % p
    return C


def mat_vec_mul(M, v, p):
    """Matrix-vector multiplication mod p."""
    t = len(M)
    result = [0] * t
    for i in range(t):
        s = 0
        for j in range(t):
            s += M[i][j] * v[j]
        result[i] = s % p
    return result


def mat_pow(M, n, p):
    """Matrix exponentiation via repeated squaring."""
    t = len(M)
    result = mat_identity(t, p)
    base = [row[:] for row in M]
    while n > 0:
        if n & 1:
            result = mat_mul(result, base, p)
        base = mat_mul(base, base, p)
        n >>= 1
    return result


def mat_add_scalar_diag(M, c, p):
    """Return M + c*I mod p."""
    t = len(M)
    R = [row[:] for row in M]
    for i in range(t):
        R[i][i] = (R[i][i] + c) % p
    return R


def mat_trace(M, p):
    """Trace of a square matrix mod p."""
    return sum(M[i][i] for i in range(len(M))) % p


def mat_sub(A, B, p):
    """Matrix subtraction mod p."""
    t = len(A)
    return [[(A[i][j] - B[i][j]) % p for j in range(t)] for i in range(t)]


# =============================================================================
# Characteristic Polynomial (Faddeev-LeVerrier Algorithm)
# =============================================================================


def char_poly(M, p):
    """
    Compute the characteristic polynomial of M over GF(p).

    Returns coefficients [c_0, c_1, ..., c_{n-1}, 1] (monic, ascending order).
    """
    n = len(M)
    coeffs = [0] * (n + 1)
    coeffs[n] = 1

    C = [row[:] for row in M]
    coeffs[n - 1] = (-mat_trace(C, p)) % p

    for k in range(2, n + 1):
        temp = mat_add_scalar_diag(C, coeffs[n - k + 1], p)
        C = mat_mul(M, temp, p)
        coeffs[n - k] = (-(mod_inv(k, p) * mat_trace(C, p)) % p) % p

    return coeffs


# =============================================================================
# Polynomial Operations over GF(p)
# =============================================================================


def poly_strip(f):
    """Remove trailing zero coefficients."""
    f = list(f)
    while len(f) > 1 and f[-1] == 0:
        f.pop()
    return f


def poly_mul(f, g, p):
    """Multiply two polynomials over GF(p)."""
    if not f or not g:
        return [0]
    n, m = len(f), len(g)
    result = [0] * (n + m - 1)
    for i in range(n):
        if f[i] == 0:
            continue
        for j in range(m):
            result[i + j] = (result[i + j] + f[i] * g[j]) % p
    return poly_strip(result)


def poly_divmod(f, g, p):
    """Polynomial division f / g over GF(p). Returns (quotient, remainder)."""
    f = list(f)
    g = poly_strip(g)
    if g == [0]:
        raise ZeroDivisionError("Division by zero polynomial")
    dg = len(g) - 1
    inv_lc = mod_inv(g[-1], p)
    q = [0] * max(len(f) - dg, 1)
    while len(f) >= len(g) and f != [0]:
        f = poly_strip(f)
        if len(f) < len(g):
            break
        coeff = (f[-1] * inv_lc) % p
        shift = len(f) - len(g)
        q[shift] = coeff
        for i in range(len(g)):
            f[shift + i] = (f[shift + i] - coeff * g[i]) % p
        f = poly_strip(f)
    return poly_strip(q), poly_strip(f) if f else [0]


def poly_mod(f, g, p):
    """Polynomial remainder f mod g over GF(p)."""
    _, r = poly_divmod(f, g, p)
    return r


def poly_gcd(f, g, p):
    """GCD of two polynomials over GF(p) via Euclidean algorithm."""
    f = poly_strip(list(f))
    g = poly_strip(list(g))
    while g != [0]:
        f, g = g, poly_mod(f, g, p)
    if len(f) > 0 and f[-1] != 0:
        inv_lc = mod_inv(f[-1], p)
        f = [(c * inv_lc) % p for c in f]
    return poly_strip(f)


def poly_pow_mod(base, exp, modulus, p):
    """Compute base^exp mod modulus in GF(p)[x] via repeated squaring."""
    if exp == 0:
        return [1]
    result = [1]
    base = poly_mod(base, modulus, p)
    while exp > 0:
        if exp & 1:
            result = poly_mod(poly_mul(result, base, p), modulus, p)
        base = poly_mod(poly_mul(base, base, p), modulus, p)
        exp >>= 1
    return result


def poly_sub(f, g, p):
    """Subtract polynomial g from f over GF(p)."""
    n = max(len(f), len(g))
    result = [0] * n
    for i in range(len(f)):
        result[i] = f[i]
    for i in range(len(g)):
        result[i] = (result[i] - g[i]) % p
    return poly_strip(result)


def poly_eval(f, x, p):
    """Evaluate polynomial f at point x over GF(p)."""
    result = 0
    for i in range(len(f) - 1, -1, -1):
        result = (result * x + f[i]) % p
    return result


def prime_factors(n):
    """Return the set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


# =============================================================================
# Root Finding over GF(p) — Cantor-Zassenhaus Algorithm
# =============================================================================


def find_roots_gfp(f, p):
    """
    Find all roots of polynomial f in GF(p).

    Uses Cantor-Zassenhaus: first extract the product of all linear factors
    via gcd(f, x^p - x), then split using random translations.

    Returns a list of roots.
    """
    if len(f) <= 1:
        return []

    # Step 1: g = gcd(f, x^p - x) — product of all distinct linear factors
    x = [0, 1]
    x_p = poly_pow_mod(x, p, f, p)
    x_p_minus_x = poly_sub(x_p, x, p)
    g = poly_gcd(f, x_p_minus_x, p)

    if len(g) <= 1:
        return []  # No roots in GF(p)

    # Step 2: Factor g into linear factors
    return _split_linear(g, p)


def _split_linear(g, p):
    """Recursively split a squarefree polynomial (all linear factors) into roots."""
    deg = len(g) - 1
    if deg == 0:
        return []
    if deg == 1:
        # g(x) = g[1]*x + g[0], root = -g[0]/g[1]
        root = ((-g[0]) * mod_inv(g[1], p)) % p
        return [root]

    # Try successive values of a for Cantor-Zassenhaus split
    half = (p - 1) // 2
    for a in range(p):
        # h(x) = (x + a)^((p-1)/2) - 1 mod g(x)
        xa = poly_strip([a % p, 1])  # x + a
        h = poly_pow_mod(xa, half, g, p)
        h = poly_sub(h, [1], p)  # h - 1

        d = poly_gcd(h, g, p)
        dd = len(d) - 1
        if 0 < dd < deg:
            # Nontrivial split
            q, _ = poly_divmod(g, d, p)
            return _split_linear(d, p) + _split_linear(q, p)

    return []  # Should not happen for squarefree polynomials


# =============================================================================
# Linear Algebra over GF(p) — Vector Spaces and Subspaces
# =============================================================================


def rref(rows, p, ncols=None):
    """
    Row-reduce a list of row vectors over GF(p) to reduced row echelon form.

    Returns the list of nonzero rows (basis vectors).
    """
    if not rows:
        return []
    if ncols is None:
        ncols = len(rows[0])
    mat = [list(row) for row in rows]
    nrows = len(mat)
    pivot_row = 0

    for col in range(ncols):
        # Find pivot
        found = -1
        for row in range(pivot_row, nrows):
            if mat[row][col] % p != 0:
                found = row
                break
        if found == -1:
            continue
        # Swap
        mat[pivot_row], mat[found] = mat[found], mat[pivot_row]
        # Scale pivot row
        inv_val = mod_inv(mat[pivot_row][col], p)
        mat[pivot_row] = [(x * inv_val) % p for x in mat[pivot_row]]
        # Eliminate
        for row in range(nrows):
            if row != pivot_row and mat[row][col] % p != 0:
                factor = mat[row][col]
                mat[row] = [(mat[row][j] - factor * mat[pivot_row][j]) % p for j in range(ncols)]
        pivot_row += 1

    return [row for row in mat if any(x % p != 0 for x in row)]


def null_space(M, p):
    """
    Compute the null space of matrix M over GF(p).

    Returns a list of basis vectors.
    """
    t = len(M)
    ncols = len(M[0]) if M else 0

    # Augment: [M | I]
    aug = [list(M[i]) + [1 if j == i else 0 for j in range(t)] for i in range(t)]
    basis = rref(aug, p, ncols=ncols + t)

    # Rows where the first ncols entries are all zero give null space vectors
    null_vecs = []
    for row in aug:
        # Re-reduce
        pass

    # Alternative: use the standard kernel computation
    # Build augmented matrix [M^T | I_ncols] and row-reduce
    # Actually, let's use a cleaner approach
    if t == 0 or ncols == 0:
        return []

    # Transpose M, row-reduce [M^T | I], extract kernel from identity part
    # Simpler: just build [M] and find vectors v such that M*v = 0

    # Gaussian elimination on M
    mat = [list(row) for row in M]
    pivots = {}  # col -> row
    nrows = len(mat)

    mat_copy = [list(row) for row in M]
    pivot_row = 0
    pivot_cols = []

    for col in range(ncols):
        found = -1
        for row in range(pivot_row, nrows):
            if mat_copy[row][col] % p != 0:
                found = row
                break
        if found == -1:
            continue
        mat_copy[pivot_row], mat_copy[found] = mat_copy[found], mat_copy[pivot_row]
        inv_val = mod_inv(mat_copy[pivot_row][col], p)
        mat_copy[pivot_row] = [(x * inv_val) % p for x in mat_copy[pivot_row]]
        for row in range(nrows):
            if row != pivot_row and mat_copy[row][col] % p != 0:
                factor = mat_copy[row][col]
                mat_copy[row] = [(mat_copy[row][j] - factor * mat_copy[pivot_row][j]) % p for j in range(ncols)]
        pivot_cols.append(col)
        pivot_row += 1

    # Free variables are columns not in pivot_cols
    free_cols = [c for c in range(ncols) if c not in pivot_cols]
    null_vecs = []

    for fc in free_cols:
        vec = [0] * ncols
        vec[fc] = 1
        for i, pc in enumerate(pivot_cols):
            vec[pc] = (-mat_copy[i][fc]) % p
        null_vecs.append(vec)

    return null_vecs


def subspace_basis(vectors, p, t):
    """Compute a basis for the subspace spanned by the given vectors over GF(p)."""
    if not vectors:
        return []
    return rref(vectors, p, ncols=t)


def subspace_intersection(basis1, basis2, p, t):
    """
    Compute the intersection of two subspaces (given as lists of basis vectors)
    over GF(p) in an ambient space of dimension t.

    Uses the standard method: if V has basis {v1,...,vk} and W has basis {w1,...,wl},
    find vectors c such that sum(c_i * v_i) = sum(d_j * w_j).
    Equivalently, find the kernel of [V^T | -W^T].
    """
    if not basis1 or not basis2:
        return []

    k = len(basis1)
    l = len(basis2)

    # Build matrix: each row is a dimension-t vector
    # We want solutions to: c_1*v_1 + ... + c_k*v_k - d_1*w_1 - ... - d_l*w_l = 0
    # This is the null space of the (k+l) x t matrix [v_1; ...; v_k; -w_1; ...; -w_l]^T
    # Or equivalently: the matrix [V | -W] where V is t x k and W is t x l

    # Build the t x (k+l) matrix
    M = [[0] * (k + l) for _ in range(t)]
    for j in range(k):
        for i in range(t):
            M[i][j] = basis1[j][i] % p
    for j in range(l):
        for i in range(t):
            M[i][k + j] = (-basis2[j][i]) % p

    # Find null space of this t x (k+l) matrix
    ns = null_space_rect(M, p, t, k + l)

    # Each null vector [c_1,...,c_k, d_1,...,d_l] gives an intersection vector
    # v = sum(c_i * v_i) = sum(d_j * w_j)
    result = []
    for nv in ns:
        vec = [0] * t
        for j in range(k):
            for i in range(t):
                vec[i] = (vec[i] + nv[j] * basis1[j][i]) % p
        if any(x != 0 for x in vec):
            result.append(vec)

    return subspace_basis(result, p, t)


def null_space_rect(M, p, nrows, ncols):
    """Compute null space of an nrows x ncols matrix over GF(p)."""
    mat = [list(M[i]) for i in range(nrows)]

    pivot_row = 0
    pivot_cols = []

    for col in range(ncols):
        found = -1
        for row in range(pivot_row, nrows):
            if mat[row][col] % p != 0:
                found = row
                break
        if found == -1:
            continue
        mat[pivot_row], mat[found] = mat[found], mat[pivot_row]
        inv_val = mod_inv(mat[pivot_row][col], p)
        mat[pivot_row] = [(x * inv_val) % p for x in mat[pivot_row]]
        for row in range(nrows):
            if row != pivot_row and mat[row][col] % p != 0:
                factor = mat[row][col]
                mat[row] = [(mat[row][j] - factor * mat[pivot_row][j]) % p for j in range(ncols)]
        pivot_cols.append(col)
        pivot_row += 1

    free_cols = [c for c in range(ncols) if c not in pivot_cols]
    null_vecs = []

    for fc in free_cols:
        vec = [0] * ncols
        vec[fc] = 1
        for i, pc in enumerate(pivot_cols):
            vec[pc] = (-mat[i][fc]) % p
        null_vecs.append(vec)

    return null_vecs


def subspace_dim(basis):
    """Dimension of a subspace = number of basis vectors."""
    return len(basis)


def subspace_eq(basis1, basis2, p, t):
    """Check if two subspaces are equal."""
    if len(basis1) != len(basis2):
        return False
    if not basis1 and not basis2:
        return True
    # V1 ⊆ V2 and V2 ⊆ V1
    combined = subspace_basis(basis1 + basis2, p, t)
    return len(combined) == len(basis1)


def full_space_basis(t):
    """Standard basis for the full t-dimensional space."""
    return [[(1 if i == j else 0) for j in range(t)] for i in range(t)]


# =============================================================================
# Grain SR-LFSR (Self-Shrinking Generator)
# =============================================================================


class GrainLFSR:
    """
    Grain SR-LFSR for deterministic pseudorandom generation.

    Implements the self-shrinking generator from Appendix E of the
    Poseidon paper (https://eprint.iacr.org/2019/458).

    Initialization seed (80 bits, big-endian):

      +---------+--------+--------+--------+--------+--------+------------+
      |field (2)|sbox (4)| n (12) | t (12) | RF(10) | RP(10) | ones (30)  |
      +---------+--------+--------+--------+--------+--------+------------+

    LFSR feedback polynomial taps: positions [0, 13, 23, 38, 51, 62]
    Burn-in: 160 clock cycles (output discarded)
    Output: self-shrinking mode — generate pairs (a, b); if a=1 output b, else discard
    """

    TAPS = [0, 13, 23, 38, 51, 62]

    def __init__(self, n, t, R_F, R_P):
        field_type = 1  # GF(p)
        sbox = 0  # x^alpha

        bits = []
        bits += self._to_bits(field_type, 2)
        bits += self._to_bits(sbox, 4)
        bits += self._to_bits(n, 12)
        bits += self._to_bits(t, 12)
        bits += self._to_bits(R_F, 10)
        bits += self._to_bits(R_P, 10)
        bits += [1] * 30
        assert len(bits) == 80

        self.state = bits

        for _ in range(160):
            self._clock()

    @staticmethod
    def _to_bits(value, width):
        return [int(b) for b in bin(value)[2:].zfill(width)]

    def _clock(self):
        new_bit = 0
        for tap in self.TAPS:
            new_bit ^= self.state[tap]
        self.state.pop(0)
        self.state.append(new_bit)
        return new_bit

    def next_bit(self):
        while True:
            a = self._clock()
            b = self._clock()
            if a == 1:
                return b

    def random_field_element(self, n, p):
        while True:
            bits = [self.next_bit() for _ in range(n)]
            value = int("".join(str(b) for b in bits), 2)
            if value < p:
                return value


# =============================================================================
# Security Analysis — Round Number Computation
# =============================================================================


def sat_inequalities(p, t, R_F, R_P, alpha, M, n):
    """
    Check whether (R_F, R_P) satisfies all security inequalities.

    Same analysis for both Poseidon and Poseidon2 (per Poseidon2 paper Section 6).
    """
    threshold = (floor(log(p, 2) - ((alpha - 1) / 2.0))) * (t + 1)
    R_F_1 = 6 if M <= threshold else 10

    R_F_2 = 1 + ceil(log(2, alpha) * min(M, n)) + ceil(log(t, alpha)) - R_P
    R_F_3 = log(2, alpha) * min(M, log(p, 2)) - R_P
    R_F_4 = t - 1 + log(2, alpha) * min(M / float(t + 1), log(p, 2) / 2.0) - R_P
    R_F_5 = (t - 2 + (M / (2.0 * log(alpha, 2))) - R_P) / float(t - 1)

    R_F_max = max(ceil(R_F_1), ceil(R_F_2), ceil(R_F_3), ceil(R_F_4), ceil(R_F_5))

    r_temp = floor(t / 3.0)
    over = (R_F - 1) * t + R_P + r_temp + r_temp * (R_F / 2.0) + R_P + alpha
    under = r_temp * (R_F / 2.0) + R_P + alpha

    try:
        from math import comb
        binom_val = comb(int(over), int(under))
        if binom_val == 0:
            binom_log = 0
        else:
            binom_log = log2(binom_val)
    except (ValueError, OverflowError):
        binom_log = M + 1

    cost_gb4 = ceil(2 * binom_log)
    return (R_F >= R_F_max) and (cost_gb4 >= M)


def compute_round_numbers(p, t, alpha, M=128):
    """Compute optimal (R_F, R_P) with security margin (+2 R_F, *1.075 R_P)."""
    n = p.bit_length()

    best_R_F = 0
    best_R_P = 0
    min_cost = float("inf")
    max_cost_rf = 0

    for R_P_t in range(1, 500):
        for R_F_t in range(4, 100):
            if R_F_t % 2 != 0:
                continue
            if sat_inequalities(p, t, R_F_t, R_P_t, alpha, M, n):
                R_F_m = R_F_t + 2
                R_P_m = int(ceil(R_P_t * 1.075))
                cost = t * R_F_m + R_P_m
                if (cost < min_cost) or (cost == min_cost and R_F_m < max_cost_rf):
                    best_R_P = R_P_m
                    best_R_F = R_F_m
                    min_cost = cost
                    max_cost_rf = best_R_F

    if best_R_F == 0:
        raise ValueError(f"No valid round numbers found for p={p}, t={t}, alpha={alpha}")

    return (best_R_F, best_R_P)


# =============================================================================
# Round Constant Generation (Poseidon1)
# =============================================================================


def generate_round_constants_poseidon1(grain, p, n, t, R_F, R_P):
    """
    Generate Poseidon1 round constants from the Grain LFSR.

    Total constants: (R_F + R_P) * t
    Every round (full and partial) has t constants.

    Returns:
        list of (R_F + R_P) lists of t constants each
    """
    num_rounds = R_F + R_P
    num_constants = num_rounds * t

    raw = []
    for _ in range(num_constants):
        raw.append(grain.random_field_element(n, p))

    # Split into per-round arrays
    rounds = []
    for r in range(num_rounds):
        rounds.append(raw[r * t : (r + 1) * t])

    return rounds


# =============================================================================
# MDS Matrix Generation (Cauchy Construction)
# =============================================================================


def generate_cauchy_mds(grain, p, n, t):
    """
    Generate a Cauchy MDS matrix using elements from the Grain LFSR.

    M[i,j] = 1 / (xs[i] + ys[j])

    where xs and ys are 2t distinct Grain-sampled field elements,
    with no pair summing to zero.
    """
    while True:
        # Sample 2t distinct elements
        rand_list = []
        seen = set()
        while len(rand_list) < 2 * t:
            val = grain.random_field_element(n, p)
            if val not in seen:
                rand_list.append(val)
                seen.add(val)

        xs = rand_list[:t]
        ys = rand_list[t:]

        # Check no xs[i] + ys[j] == 0 mod p
        valid = True
        for i in range(t):
            for j in range(t):
                if (xs[i] + ys[j]) % p == 0:
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            continue

        # Build Cauchy matrix
        M = [[0] * t for _ in range(t)]
        for i in range(t):
            for j in range(t):
                M[i][j] = mod_inv((xs[i] + ys[j]) % p, p)

        return M


# =============================================================================
# MDS Security Verification — Algorithms 1, 2, 3
#
# From [GRS21] — Grassi, Rechberger, Schofnegger:
# "Proving Resistance Against Infinitely Long Subspace Trails"
# =============================================================================


def generate_vectorspace(round_num, M, M_round, t, p):
    """
    Generate the vectorspace S for a given round in Algorithm 1.

    S_0 = full space
    S_1 = span of basis vectors e_1, ..., e_{t-1} (all except e_0)
    S_i = {v : v = [0]*s ++ w, w in right_kernel of truncated M_round matrix}
    """
    s = 1
    full_basis = full_space_basis(t)

    if round_num == 0:
        return full_basis
    elif round_num == 1:
        return full_basis[s:]
    else:
        # Build the matrix from truncated rows of M_round[0..round_num-1]
        rows = []
        for i in range(round_num - 1):
            # Take first s rows of M_round[i], columns s..t-1
            for j in range(s):
                row = [M_round[i][j][c] for c in range(s, t)]
                rows.append(row)

        if not rows:
            return full_basis[s:]

        # Compute right kernel
        nrows = len(rows)
        ncols = t - s
        kernel = null_space_rect(
            [rows[i] if i < nrows else [0] * ncols for i in range(nrows)],
            p, nrows, ncols
        )

        # Extend kernel vectors with s leading zeros
        extended = []
        for vec in kernel:
            extended.append([0] * s + vec)

        if not extended:
            return []
        return subspace_basis(extended, p, t)


def subspace_times_matrix(basis, M, p, t):
    """Compute {M*v : v in basis} and return the resulting subspace basis."""
    new_vecs = []
    for v in basis:
        new_vecs.append(mat_vec_mul(M, v, p))
    return subspace_basis(new_vecs, p, t)


def eigenspaces_in_gfp(M, p, t):
    """
    Compute eigenspaces of M that have eigenvalues in GF(p).

    Returns list of (eigenvalue, basis_of_eigenspace) pairs.
    """
    cp = char_poly(M, p)
    roots = find_roots_gfp(cp, p)

    result = []
    for lam in roots:
        # Eigenspace = null space of (M - lambda*I)
        M_shifted = [row[:] for row in M]
        for i in range(t):
            M_shifted[i][i] = (M_shifted[i][i] - lam) % p
        # Null space
        ns = null_space_rect(M_shifted, p, t, t)
        if ns:
            result.append((lam, ns))

    return result


def algorithm_1(M, t, p):
    """
    Algorithm 1 from [GRS21]: Check for circulant and eigenspace weaknesses.

    Returns True if the matrix is considered secure.
    """
    s = 1
    r = floor((t - s) / float(s))

    M_round = []
    for j in range(t + 1):
        M_round.append(mat_pow(M, j + 1, p))

    for i in range(1, int(r) + 1):
        mat_test = mat_pow(M, i, p)

        # Check if M^i is a scalar multiple of identity (circulant check)
        entry = mat_test[0][0]
        is_circulant = True
        for row in range(t):
            for col in range(t):
                expected = entry if row == col else 0
                if mat_test[row][col] % p != expected % p:
                    is_circulant = False
                    break
            if not is_circulant:
                break
        if is_circulant:
            return False

        S = generate_vectorspace(i, M, M_round, t, p)

        # Compute eigenspaces in GF(p) and intersect with S
        espaces = eigenspaces_in_gfp(mat_test, p, t)
        all_intersection_vecs = []
        for _lam, ebasis in espaces:
            inter = subspace_intersection(S, ebasis, p, t)
            all_intersection_vecs.extend(inter)

        IS_basis = subspace_basis(all_intersection_vecs, p, t)

        if subspace_dim(IS_basis) >= 1 and not subspace_eq(IS_basis, full_space_basis(t), p, t):
            return False

        for j in range(1, i + 1):
            S_mul = subspace_times_matrix(S, mat_pow(M, j, p), p, t)
            if subspace_eq(S, S_mul, p, t):
                return False

    return True


def algorithm_2(M, t, p):
    """
    Algorithm 2 from [GRS21]: Check subspace trail resistance.

    Returns True if the matrix is considered secure.
    """
    s = 1
    full_basis = full_space_basis(t)

    # I_powerset = all nonempty subsets of {0, ..., s-1}
    # For s=1, this is just [{0}]
    I_subsets = [[0]]

    for I_s in I_subsets:
        test_next = False
        # IS = span of {e_i : i in I_s}
        IS_basis = [full_basis[i] for i in I_s]
        # full_iota_space = span of {e_i : i in I_s} + {e_i : i >= s}
        full_iota_vecs = [full_basis[i] for i in I_s] + full_basis[s:]
        full_iota_basis = subspace_basis(full_iota_vecs, p, t)

        for l in I_s:
            v = list(full_basis[l])
            while True:
                delta = subspace_dim(IS_basis)
                v = mat_vec_mul(M, v, p)
                IS_basis = subspace_basis(IS_basis + [v], p, t)

                if subspace_dim(IS_basis) == t:
                    test_next = True
                    break
                inter = subspace_intersection(IS_basis, full_iota_basis, p, t)
                if not subspace_eq(inter, IS_basis, p, t):
                    test_next = True
                    break
                if subspace_dim(IS_basis) <= delta:
                    break
            if test_next:
                break
        if test_next:
            continue
        return False

    return True


def algorithm_3(M, t, p):
    """
    Algorithm 3 from [GRS21]: Check higher-order subspace trail resistance.

    Returns True if the matrix is considered secure.
    """
    l = 4 * t
    for r in range(2, l + 1):
        M_r = mat_pow(M, r, p)
        if not algorithm_2(M_r, t, p):
            return False
    return True


def generate_secure_mds(grain, p, n, t, verbose=False):
    """
    Generate an MDS matrix that passes all three security algorithms.

    Retries with fresh Grain samples until all checks pass.
    """
    attempt = 0
    while True:
        attempt += 1
        mds = generate_cauchy_mds(grain, p, n, t)
        if verbose and attempt % 10 == 0:
            print(f"  MDS attempt {attempt}...", flush=True)

        r1 = algorithm_1(mds, t, p)
        if not r1:
            continue
        r2 = algorithm_2(mds, t, p)
        if not r2:
            continue
        r3 = algorithm_3(mds, t, p)
        if not r3:
            continue

        if verbose:
            print(f"  Secure MDS matrix found after {attempt} attempt(s)")
        return mds


# =============================================================================
# Poseidon1 Reference Permutation
# =============================================================================


def poseidon1_permutation(state, mds, round_constants, alpha, p, t, R_F, R_P):
    """
    Reference implementation of the Poseidon (original) permutation.

    All rounds use the same MDS matrix and t round constants.
    Full rounds apply S-box to all elements; partial rounds only to s0.
    """
    state = list(state)
    R_f = R_F // 2
    rc_idx = 0

    # First R_f full rounds
    for _ in range(R_f):
        for i in range(t):
            state[i] = (state[i] + round_constants[rc_idx][i]) % p
        for i in range(t):
            state[i] = pow(state[i], alpha, p)
        state = mat_vec_mul(mds, state, p)
        rc_idx += 1

    # R_P partial rounds
    for _ in range(R_P):
        for i in range(t):
            state[i] = (state[i] + round_constants[rc_idx][i]) % p
        state[0] = pow(state[0], alpha, p)
        state = mat_vec_mul(mds, state, p)
        rc_idx += 1

    # Last R_f full rounds
    for _ in range(R_f):
        for i in range(t):
            state[i] = (state[i] + round_constants[rc_idx][i]) % p
        for i in range(t):
            state[i] = pow(state[i], alpha, p)
        state = mat_vec_mul(mds, state, p)
        rc_idx += 1

    return state


# =============================================================================
# Output Formatting
# =============================================================================


def format_hex(value, n):
    """Format a field element as a hex string with appropriate width."""
    hex_width = (n + 3) // 4
    return f"0x{value:0{hex_width}x}"


def _wrap_hex_row(values, n, indent=4, max_width=100):
    """Wrap a list of hex values into multiple lines if they exceed max_width."""
    items = [format_hex(v, n) for v in values]
    prefix = " " * indent
    lines = []
    current = prefix
    for i, item in enumerate(items):
        sep = "  " if i > 0 else ""
        if len(current) + len(sep) + len(item) > max_width and current.strip():
            lines.append(current)
            current = prefix + item
        else:
            current += sep + item
    if current.strip():
        lines.append(current)
    return lines


def format_default_poseidon1(
    field_name, width, round_constants, mds, p, n, alpha, R_F, R_P, skip_mds,
):
    """
    Format Poseidon1 constants as a clean, language-neutral summary.

    This is the default output: human-readable, no language-specific syntax.
    """
    R_f = R_F // 2
    num_rounds = R_F + R_P
    sep = "─" * 72
    lines = []

    lines.append(sep)
    lines.append(f"  Poseidon Constants — {field_name} (width {width})")
    lines.append(sep)
    lines.append("")
    lines.append(f"  Field        {field_name}")
    lines.append(f"  Prime (p)    {p}")
    lines.append(f"  Bit length   {n}")
    lines.append(f"  S-box (α)    x^{alpha}")
    lines.append(f"  Width (t)    {width}")
    lines.append(f"  Full rounds  {R_F}  ({R_f} initial + {R_f} final)")
    lines.append(f"  Partial      {R_P}")
    lines.append(f"  Total rounds {num_rounds}")
    lines.append(f"  Constants    {num_rounds * width}  ({num_rounds} × {width})")
    lines.append("")

    # Round constants — first R_f full rounds
    lines.append(sep)
    lines.append(f"  Round Constants — Full Initial ({R_f} rounds × {width})")
    lines.append(sep)
    for i in range(R_f):
        lines.append("")
        lines.append(f"  round {i}:")
        lines.extend(_wrap_hex_row(round_constants[i], n))

    lines.append("")

    # Partial rounds
    lines.append(sep)
    lines.append(f"  Round Constants — Partial ({R_P} rounds × {width})")
    lines.append(sep)
    for i in range(R_P):
        lines.append("")
        lines.append(f"  round {R_f + i}:")
        lines.extend(_wrap_hex_row(round_constants[R_f + i], n))

    lines.append("")

    # Final full rounds
    lines.append(sep)
    lines.append(f"  Round Constants — Full Final ({R_f} rounds × {width})")
    lines.append(sep)
    for i in range(R_f):
        lines.append("")
        lines.append(f"  round {R_f + R_P + i}:")
        lines.extend(_wrap_hex_row(round_constants[R_f + R_P + i], n))

    # MDS matrix
    if not skip_mds:
        lines.append("")
        lines.append(sep)
        lines.append(f"  MDS Matrix ({width} × {width})")
        lines.append(sep)
        for i, row in enumerate(mds):
            lines.append("")
            lines.append(f"  row {i}:")
            lines.extend(_wrap_hex_row(row, n))

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)



def format_json_poseidon1(
    field_name, width, round_constants, mds, p, n, alpha, R_F, R_P, skip_mds,
):
    """Format Poseidon1 constants as JSON."""
    data = {
        "field": field_name,
        "prime": str(p),
        "width": width,
        "alpha": alpha,
        "R_F": R_F,
        "R_P": R_P,
        "round_constants": [[format_hex(v, n) for v in rnd] for rnd in round_constants],
    }
    if not skip_mds:
        data["mds_matrix"] = [[format_hex(v, n) for v in row] for row in mds]
    return json.dumps(data, indent=2)


# =============================================================================
# Rust Constant Verification
# =============================================================================


def _parse_rust_round_constants(rust_path, const_name, width):
    """
    Parse a Poseidon1 2D round-constant array from a Rust source file.

    Returns:
        List[List[int]] with shape [num_rounds][width].
    """
    source = rust_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"pub const {re.escape(const_name)}\s*:\s*\[\[[^\]]+;\s*{width}\s*\];\s*(\d+)\s*\]\s*="
        rf"\s*[^=]+?::new_2d_array\(\[(.*?)\]\);",
        re.S,
    )
    match = pattern.search(source)
    if not match:
        raise ValueError(
            f"Could not find constant '{const_name}' with width {width} in {rust_path}"
        )

    declared_rounds = int(match.group(1))
    body = match.group(2)

    # Strip line comments so numeric extraction isn't polluted by doc examples.
    body = "\n".join(line.split("//", 1)[0] for line in body.splitlines())

    # Collect top-level rows from the 2D array payload.
    rows = []
    depth = 0
    current = []
    for ch in body:
        if ch == "[":
            depth += 1
            if depth == 1:
                current = []
                continue
        if ch == "]":
            if depth == 1:
                rows.append("".join(current))
                current = []
                depth -= 1
                continue
            depth -= 1
        if depth >= 1:
            current.append(ch)

    parsed = []
    for row in rows:
        tokens = re.findall(r"0x[0-9a-fA-F_]+|\d+", row)
        values = [
            int(tok.replace("_", ""), 16) if tok.lower().startswith("0x") else int(tok)
            for tok in tokens
        ]
        parsed.append(values)

    if len(parsed) != declared_rounds:
        raise ValueError(
            f"{const_name} in {rust_path} declares {declared_rounds} rounds but parsed {len(parsed)}"
        )
    for idx, values in enumerate(parsed):
        if len(values) != width:
            raise ValueError(
                f"{const_name} row {idx} in {rust_path} has width {len(values)}, expected {width}"
            )
    return parsed


def verify_generated_constants_against_rust(field_name, width, generated_round_constants, repo_root=None):
    """
    Compare freshly generated constants with in-tree Rust constants.

    Returns:
        (ok: bool, message: str)
    """
    key = (field_name, width)
    if key not in RUST_ROUND_CONSTANTS:
        return False, f"No in-tree mapping defined for field={field_name}, width={width}"

    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    rel_path, const_name = RUST_ROUND_CONSTANTS[key]
    rust_path = root / rel_path
    if not rust_path.exists():
        return False, f"Rust file not found: {rust_path}"

    rust_constants = _parse_rust_round_constants(rust_path, const_name, width)

    if len(rust_constants) != len(generated_round_constants):
        return (
            False,
            f"Round count mismatch for {const_name}: rust={len(rust_constants)} "
            f"generated={len(generated_round_constants)}",
        )

    for round_idx, (rust_row, gen_row) in enumerate(zip(rust_constants, generated_round_constants)):
        if len(rust_row) != len(gen_row):
            return (
                False,
                f"Width mismatch at round {round_idx} for {const_name}: "
                f"rust={len(rust_row)} generated={len(gen_row)}",
            )
        for col_idx, (rust_v, gen_v) in enumerate(zip(rust_row, gen_row)):
            if rust_v != gen_v:
                return (
                    False,
                    f"Mismatch in {const_name} at round {round_idx}, col {col_idx}: "
                    f"rust={hex(rust_v)} generated={hex(gen_v)}",
                )

    return True, f"Rust constants match generated values: {const_name}"


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate Poseidon (original) round constants for various prime fields.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python poseidon/generate_constants.py --field babybear --width 16\n"
               "  python poseidon/generate_constants.py --field goldilocks --width 8 --format json\n"
               "  python poseidon/generate_constants.py --field koalabear --width 24 -v\n",
    )
    parser.add_argument(
        "--field", required=True,
        choices=list(FIELDS.keys()),
        help="Target field",
    )
    parser.add_argument(
        "--width", required=True, type=int,
        help="State width (t)",
    )
    parser.add_argument(
        "--format", default="default",
        choices=["default", "json"],
        help="Output format (default: human-readable summary)",
    )
    parser.add_argument(
        "--security-level", default=128, type=int,
        help="Security level in bits (default: 128)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print verbose progress information",
    )
    parser.add_argument(
        "--skip-mds", action="store_true",
        help="Skip MDS matrix generation and security checks (much faster)",
    )
    parser.add_argument(
        "--test-vector", action="store_true",
        help="Compute and print a test vector using the reference permutation",
    )
    parser.add_argument(
        "--check-rust", action="store_true",
        help="Verify generated round constants against in-tree Rust constants for this field/width",
    )
    parser.add_argument(
        "--check-rust-only", action="store_true",
        help="Run --check-rust and suppress formatted constant output",
    )
    parser.add_argument(
        "--repo-root", default=None,
        help="Repository root to use for --check-rust (defaults to script-relative root)",
    )
    args = parser.parse_args()
    if args.check_rust_only:
        args.check_rust = True

    field_info = FIELDS[args.field]
    p = field_info["prime"]
    t = args.width

    if t not in field_info["valid_widths"]:
        print(f"Error: width {t} not valid for {args.field}. "
              f"Valid widths: {field_info['valid_widths']}", file=sys.stderr)
        sys.exit(1)

    n = p.bit_length()
    alpha = compute_alpha(p)

    if args.verbose:
        print(f"Field: {args.field}")
        print(f"Prime: {p} ({n} bits)")
        print(f"Width: {t}")
        print(f"Alpha: {alpha}")
        print()

    # --- Compute round numbers ---
    if args.verbose:
        print("Computing round numbers...", flush=True)
    R_F, R_P = compute_round_numbers(p, t, alpha, args.security_level)
    if args.verbose:
        print(f"R_F = {R_F}, R_P = {R_P}")
        print(f"Total constants: (R_F + R_P) * t = {(R_F + R_P) * t}")
        print()

    # --- Initialize Grain LFSR ---
    grain = GrainLFSR(n, t, R_F, R_P)

    # --- Generate round constants ---
    if args.verbose:
        print("Generating round constants...", flush=True)
    round_constants = generate_round_constants_poseidon1(grain, p, n, t, R_F, R_P)

    # --- Optional in-tree Rust constant verification ---
    if args.check_rust:
        ok, msg = verify_generated_constants_against_rust(
            args.field, t, round_constants, repo_root=args.repo_root,
        )
        if not ok:
            print(f"Rust constant verification failed: {msg}", file=sys.stderr)
            sys.exit(1)
        if args.verbose:
            print(msg)

    # --- Generate MDS matrix ---
    if not args.skip_mds:
        if args.verbose:
            print("Generating MDS matrix (this may take a while for large widths)...", flush=True)
        mds = generate_secure_mds(grain, p, n, t, verbose=args.verbose)
    else:
        mds = [[0] * t for _ in range(t)]
        if args.verbose:
            print("Skipping MDS matrix generation")

    if args.verbose:
        print()

    # --- Output ---
    fmt_args = (
        args.field, t, round_constants, mds, p, n, alpha, R_F, R_P, args.skip_mds,
    )
    if not args.check_rust_only:
        if args.format == "default":
            print(format_default_poseidon1(*fmt_args))
        elif args.format == "json":
            print(format_json_poseidon1(*fmt_args))

    # --- Test vector ---
    if args.test_vector:
        if args.skip_mds:
            print("\nWarning: test vector uses zero MDS matrix (--skip-mds). "
                  "Output will NOT be meaningful.", file=sys.stderr)
        state_in = list(range(t))
        state_out = poseidon1_permutation(
            state_in, mds, round_constants, alpha, p, t, R_F, R_P,
        )
        print()
        print(f"Test vector (input = [0, 1, ..., {t-1}]):")
        print(f"  Input:  [{', '.join(format_hex(v, n) for v in state_in)}]")
        print(f"  Output: [{', '.join(format_hex(v, n) for v in state_out)}]")


if __name__ == "__main__":
    main()
