#!/usr/bin/env python3
"""
Poseidon2 Round Constant Generator

Generates round constants, external/internal matrices, and (with `--test-vector`)
embedded canonical permutation inputs/outputs.

Reference: Grassi et al., "Poseidon2: A Faster Version of the Poseidon Hash
Function" (https://eprint.iacr.org/2023/323)

The key differences between Poseidon2 and Poseidon (original) are:
  1. An initial linear layer before the first external round
  2. Two different mixing matrices: M_E (external) and M_I (internal)
  3. Only one round constant per internal round (applied to state[0])

Round structure (for width t, R_F full rounds, R_P partial rounds):

  Input → M_E → [R_F/2 external rounds] → [R_P internal rounds] → [R_F/2 external rounds] → Output

  External round: AddRoundConstants(all t) → S-box(all t) → M_E
  Internal round: AddRoundConstant(s0 only) → S-box(s0 only) → M_I

Supported fields:
  Field       Prime                  Bits  Alpha  Widths
  BabyBear    2013265921             31    7      16, 24, 32
  KoalaBear   2130706433             31    3      16, 24, 32
  Goldilocks  2^64 - 2^32 + 1        64    7      8, 12, 16, 20
  Mersenne31  2^31 - 1               31    5      16, 24, 32

Usage:
  python poseidon2/generate_constants.py --field babybear --width 16
  python poseidon2/generate_constants.py --field mersenne31 --width 32 --format json
  python poseidon2/generate_constants.py --field goldilocks --width 8 -v
  python poseidon2/generate_constants.py --field babybear --width 16 --test-vector
"""

import argparse
import hashlib
import json
import sys
from math import ceil, floor, gcd, log, log2

# =============================================================================
# Field Definitions
# =============================================================================

FIELDS = {
    "babybear": {
        "prime": 2013265921,
        "valid_widths": [16, 24, 32],
    },
    "koalabear": {
        "prime": 2130706433,
        "valid_widths": [16, 24, 32],
    },
    "goldilocks": {
        "prime": (1 << 64) - (1 << 32) + 1,
        "valid_widths": [8, 12, 16, 20],
    },
    "mersenne31": {
        "prime": (1 << 31) - 1,
        "valid_widths": [16, 24, 32],
    },
}


# Default Poseidon2 permutation test vectors (canonical literals).
DEFAULT_POSEIDON2_TEST_VECTORS = {
    ("babybear", 16): {
        "permutation": "default_babybear_poseidon2_16()",
        "rust_tests": ["baby-bear/src/poseidon2.rs — test_default_babybear_poseidon2_width_16"],
        "input": [
            894848333,
            1437655012,
            1200606629,
            1690012884,
            71131202,
            1749206695,
            1717947831,
            120589055,
            19776022,
            42382981,
            1831865506,
            724844064,
            171220207,
            1299207443,
            227047920,
            1783754913,
        ],
        "expected": [
            516096821,
            90309867,
            1101817252,
            1660784290,
            360715097,
            1789519026,
            1788910906,
            563338433,
            319524748,
            1741414159,
            1650859320,
            894311162,
            1121347488,
            1692793758,
            1052633829,
            1344246938,
        ],
    },
    ("babybear", 24): {
        "permutation": "default_babybear_poseidon2_24()",
        "rust_tests": ["baby-bear/src/poseidon2.rs — test_default_babybear_poseidon2_width_24"],
        "input": [
            886409618,
            1327899896,
            1902407911,
            591953491,
            648428576,
            1844789031,
            1198336108,
            355597330,
            1799586834,
            59617783,
            790334801,
            1968791836,
            559272107,
            31054313,
            1042221543,
            474748436,
            135686258,
            263665994,
            1962340735,
            1741539604,
            2026927696,
            449439011,
            1131357108,
            50869465,
        ],
        "expected": [
            882297297,
            1264077610,
            512812497,
            782602970,
            867738552,
            1251075457,
            309180082,
            340784773,
            524041877,
            351272188,
            404451680,
            15001466,
            322926653,
            1773004150,
            1718440818,
            674682955,
            1154713225,
            1719133502,
            324232301,
            1005243141,
            443371079,
            268735940,
            770060019,
            718377682,
        ],
    },
    ("babybear", 32): {
        "permutation": "default_babybear_poseidon2_32()",
        "rust_tests": [
            "baby-bear/src/poseidon2.rs — test_default_babybear_poseidon2_width_32 "
        ],
        "input": [
            377682961,
            1957793603,
            980981814,
            6565119,
            1583211709,
            176593168,
            1672635515,
            226854190,
            1096634172,
            1317773742,
            1472230830,
            1621534427,
            559807320,
            1484241910,
            1847825942,
            3491998,
            950152427,
            1935451636,
            275759400,
            227625951,
            1271142011,
            1492341973,
            1502961189,
            147694103,
            1939834518,
            1449972249,
            1822424048,
            1518111482,
            714203295,
            383863563,
            411489861,
            1253612091,
        ],
        "expected": [
            303440672,
            985419733,
            780962554,
            1395263823,
            188752116,
            1348917749,
            677984158,
            667170017,
            97281439,
            178741618,
            1770541242,
            1894441262,
            847173187,
            1374453653,
            1242356754,
            1485142795,
            1019698843,
            334329175,
            540395852,
            918117757,
            1288401072,
            508687761,
            996827321,
            1660764537,
            546969284,
            1848510002,
            334793951,
            736596659,
            1928951999,
            1444080616,
            55017699,
            1832626373,
        ],
    },
    ("koalabear", 16): {
        "permutation": "default_koalabear_poseidon2_16()",
        "rust_tests": ["koala-bear/src/poseidon2.rs — test_default_koalabear_poseidon2_width_16"],
        "input": [
            894848333,
            1437655012,
            1200606629,
            1690012884,
            71131202,
            1749206695,
            1717947831,
            120589055,
            19776022,
            42382981,
            1831865506,
            724844064,
            171220207,
            1299207443,
            227047920,
            1783754913,
        ],
        "expected": [
            1934285469,
            604889435,
            133449501,
            1026180808,
            1830659359,
            176667110,
            1391183747,
            351743874,
            1238264085,
            1292768839,
            2023573270,
            1201586780,
            1360691759,
            1230682461,
            748270449,
            651545025,
        ],
    },
    ("koalabear", 24): {
        "permutation": "default_koalabear_poseidon2_24()",
        "rust_tests": ["koala-bear/src/poseidon2.rs — test_default_koalabear_poseidon2_width_24"],
        "input": [
            886409618,
            1327899896,
            1902407911,
            591953491,
            648428576,
            1844789031,
            1198336108,
            355597330,
            1799586834,
            59617783,
            790334801,
            1968791836,
            559272107,
            31054313,
            1042221543,
            474748436,
            135686258,
            263665994,
            1962340735,
            1741539604,
            2026927696,
            449439011,
            1131357108,
            50869465,
        ],
        "expected": [
            382801106,
            82839311,
            1503190615,
            1987418517,
            854076995,
            1862291425,
            262755189,
            1050814217,
            722724562,
            741265943,
            1026879332,
            754316749,
            1966025564,
            1518878196,
            502200188,
            1368172258,
            845459257,
            1711434837,
            724453836,
            171032289,
            655223446,
            1098636135,
            407832555,
            1707498914,
        ],
    },
    ("koalabear", 32): {
        "permutation": "default_koalabear_poseidon2_32()",
        "rust_tests": [
            "koala-bear/src/poseidon2.rs — test_default_koalabear_poseidon2_width_32 "
        ],
        "input": [
            377639580,
            1129436247,
            1046213469,
            1189442335,
            766997073,
            331472151,
            734344924,
            499580178,
            371511009,
            1784992949,
            961094784,
            2047061722,
            1120236986,
            1332020114,
            1511787480,
            1290378453,
            1414897608,
            641041795,
            1940105940,
            1813107966,
            1798618911,
            1941729996,
            1148636543,
            505212370,
            1519289406,
            567500757,
            728728142,
            1833845584,
            1298210282,
            41111765,
            297995683,
            1596253449,
        ],
        "expected": [
            1359114333,
            192817145,
            2112759047,
            1534272756,
            262772033,
            1605905052,
            1578475422,
            1405808516,
            1637426946,
            1738584472,
            1537483685,
            1201015772,
            472885949,
            923753225,
            1756848188,
            1560950302,
            672658610,
            1934876055,
            229950235,
            798187377,
            1626970896,
            278337851,
            1054262154,
            1192644396,
            257269960,
            1845599185,
            489110817,
            1514396648,
            345626239,
            888828773,
            1894876982,
            500295195,
        ],
    },
    ("mersenne31", 16): {
        "permutation": "default_mersenne31_poseidon2_16()",
        "rust_tests": ["mersenne-31/src/poseidon2.rs — test_default_mersenne31_poseidon2_width_16"],
        "input": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "expected": [
            0x0B2C803A,
            0x5B1EE4D1,
            0x49C6B1E3,
            0x2CDC280C,
            0x310A60C8,
            0x530A729E,
            0x4E61BCB4,
            0x2E84D3C3,
            0x58709C08,
            0x7E82AC42,
            0x2162BCEF,
            0x6D153AB6,
            0x742CF0E3,
            0x2F21632D,
            0x61ADCE1E,
            0x1973D6F1,
        ],
    },
    ("mersenne31", 24): {
        "permutation": "default_mersenne31_poseidon2_24()",
        "rust_tests": ["mersenne-31/src/poseidon2.rs — test_default_mersenne31_poseidon2_width_24"],
        "input": list(range(24)),
        "expected": [
            0x2040F051,
            0x7261DBFA,
            0x4FBD519E,
            0x2320ECAF,
            0x039EF27C,
            0x48D60AD5,
            0x73CA17FF,
            0x6023111A,
            0x6C5E31E7,
            0x373CD90D,
            0x75A3AE11,
            0x00ECC878,
            0x33A7C097,
            0x244C2171,
            0x7552A38E,
            0x58D20817,
            0x00FEECB7,
            0x47C43C88,
            0x30D3001C,
            0x24D09BA6,
            0x71F241D9,
            0x1C72AB2E,
            0x4749F79D,
            0x61FF7579,
        ],
    },
    ("mersenne31", 32): {
        "permutation": "default_mersenne31_poseidon2_32()",
        "rust_tests": [
            "mersenne-31/src/poseidon2.rs — test_default_mersenne31_poseidon2_width_32"
        ],
        "input": [
            886409618,
            1327899896,
            1902407911,
            591953491,
            648428576,
            1844789031,
            1198336108,
            355597330,
            1799586834,
            59617783,
            790334801,
            1968791836,
            559272107,
            31054313,
            1042221543,
            474748436,
            135686258,
            263665994,
            1962340735,
            1741539604,
            2026927696,
            449439011,
            1131357108,
            50869465,
            894848333,
            1437655012,
            1200606629,
            1690012884,
            71131202,
            1749206695,
            1717947831,
            120589055,
        ],
        "expected": [
            1856060025,
            1254059276,
            2099136415,
            1891507371,
            202832695,
            754761125,
            1546769253,
            2039240755,
            969633288,
            117763588,
            624654727,
            1034887750,
            898944818,
            1818019588,
            1662865566,
            1426397765,
            102254187,
            1541093348,
            280956251,
            1038202157,
            1207554722,
            1615928492,
            2099241528,
            997904479,
            621678012,
            724483212,
            1292553224,
            1107946119,
            1584500975,
            1889218820,
            432786428,
            1331980049,
        ],
    },
    ("goldilocks", 8): {
        "permutation": "default_goldilocks_poseidon2_8()",
        "rust_tests": ["goldilocks/src/poseidon2.rs — test_default_goldilocks_poseidon2_width_8"],
        "input": [0, 1, 2, 3, 4, 5, 6, 7],
        "expected": [
            0x020CF04A1B214D14,
            0x84E14AAAEACAED25,
            0x1AE0F640E81C7457,
            0xA4D204CBAEB0D8A5,
            0x0CF637B627B3A7FF,
            0x788D304D948B486B,
            0x7327133EA1949AF4,
            0xF415ABB924DA395B,
        ],
    },
    ("goldilocks", 12): {
        "permutation": "default_goldilocks_poseidon2_12()",
        "rust_tests": ["goldilocks/src/poseidon2.rs — test_default_goldilocks_poseidon2_width_12"],
        "input": list(range(12)),
        "expected": [
            0xF292AB67C0F14B03,
            0x0A32F1B37656544C,
            0x053C61AB895498DE,
            0x02FF92E55B196FFB,
            0x58176E8F6F58CAB2,
            0xB0AA1206E7AEC0F8,
            0xE90C13F3DCE83CA4,
            0xF4DA15333EDF39C2,
            0x23B701C053C2CA6C,
            0xD233D593DCDFBF58,
            0x4EFFA5F9516FB52E,
            0x0AAF4489F1F40166,
        ],
    },
    ("goldilocks", 16): {
        "permutation": "default_goldilocks_poseidon2_16()",
        "rust_tests": [
            "goldilocks/src/poseidon2.rs — test_default_goldilocks_poseidon2_width_16 "
        ],
        "input": [
            5566333139537447289,
            6332619238598470014,
            6287757933960378228,
            3590347415306286571,
            5292348457121014769,
            3628350668046320542,
            12106279625554848022,
            9881167366883827448,
            11610943711509296728,
            15458533499329302584,
            9940874769430822472,
            2449195810349960580,
            6271849099249858303,
            5098003553882423470,
            2550474722356913284,
            16793706415574197586,
        ],
        "expected": [
            17363471649496040234,
            17379317055181730139,
            15593185324171680472,
            3827176038427840937,
            11044870504971387999,
            5115049712654183786,
            5044454358330956064,
            14234749237683052767,
            13255128155875443356,
            6663913933593108421,
            17059345908371998855,
            15316992698834344060,
            3568653445613141515,
            12911066927142911650,
            10746225424367601145,
            16930547820292837221,
        ],
    },
    ("goldilocks", 20): {
        "permutation": "default_goldilocks_poseidon2_20()",
        "rust_tests": [
            "goldilocks/src/poseidon2.rs — test_default_goldilocks_poseidon2_width_20"
        ],
        "input": [
            2053695854357871005,
            13679192365072849617,
            4517457392071889495,
            2574020394472462046,
            1890702223848595625,
            13662908291426823533,
            10060236952204337488,
            10892664235628797826,
            586287033698423193,
            1728372192399379054,
            4291835990902352011,
            11105285438068160209,
            10353144037217341363,
            13208230535885162025,
            12937162279754847113,
            7738774760351418614,
            8286444275301796832,
            5131712758418120108,
            16035760590688802187,
            13997525239209781023,
        ],
        "expected": [
            10059710073421003335,
            1862996910857720382,
            8932260229380488581,
            18386016385372684532,
            12311265669759287563,
            613393036196829436,
            6390432853088572721,
            5615514072753685340,
            3513888109347585023,
            16050694085256412340,
            2533165127685758232,
            7239931544609607180,
            11353919432250627885,
            15993300283351933125,
            7435802096539309142,
            17341865001684304279,
            1878126664879156972,
            17526369362932644128,
            3441129794730119380,
            5126279937009158560,
        ],
    },
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


# =============================================================================
# Characteristic Polynomial (Faddeev-LeVerrier Algorithm)
# =============================================================================


def char_poly(M, p):
    """
    Compute the characteristic polynomial of M over GF(p) using the
    Faddeev-LeVerrier algorithm.

    Returns coefficients [c_0, c_1, ..., c_{n-1}, 1] (monic, ascending order)
    where det(xI - M) = x^n + c_{n-1}*x^{n-1} + ... + c_0.
    """
    n = len(M)
    coeffs = [0] * (n + 1)
    coeffs[n] = 1

    # C_1 = M, c_{n-1} = -tr(M)
    C = [row[:] for row in M]
    coeffs[n - 1] = (-mat_trace(C, p)) % p

    for k in range(2, n + 1):
        # C_k = M * (C_{k-1} + c_{n-k+1} * I)
        temp = mat_add_scalar_diag(C, coeffs[n - k + 1], p)
        C = mat_mul(M, temp, p)
        coeffs[n - k] = (-(mod_inv(k, p) * mat_trace(C, p)) % p) % p

    return coeffs


# =============================================================================
# Polynomial Operations over GF(p)
# =============================================================================


def poly_strip(f):
    """Remove trailing zero coefficients (leading zeros in degree)."""
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
    # Normalize to monic
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


def is_irreducible(f, p):
    """
    Check if polynomial f is irreducible over GF(p).

    Uses the standard algorithm:
      1. Verify x^(p^n) = x (mod f(x))
      2. Verify gcd(x^(p^(n/q)) - x, f(x)) = 1 for each prime factor q of n
    """
    n = len(f) - 1  # degree
    if n <= 0:
        return False
    if n == 1:
        return True

    x = [0, 1]  # the polynomial x

    # Compute x^(p^i) mod f iteratively using Frobenius endomorphism
    # x_pi[i] = x^(p^i) mod f(x)
    x_pi = [None] * (n + 1)
    x_pi[0] = x
    for i in range(1, n + 1):
        x_pi[i] = poly_pow_mod(x_pi[i - 1], p, f, p)

    # Check 1: x^(p^n) = x (mod f)
    diff = poly_sub(x_pi[n], x, p)
    if poly_mod(diff, f, p) != [0]:
        return False

    # Check 2: for each prime factor q of n, gcd(x^(p^(n/q)) - x, f) = [1]
    for q in prime_factors(n):
        k = n // q
        diff = poly_sub(x_pi[k], x, p)
        g = poly_gcd(diff, f, p)
        if len(g) > 1:  # gcd has degree >= 1, so f is reducible
            return False

    return True


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
        """
        Initialize the Grain LFSR with Poseidon parameters.

        Args:
            n: Field size in bits
            t: State width
            R_F: Number of full rounds
            R_P: Number of partial rounds
        """
        field_type = 1  # GF(p)
        sbox = 0  # x^alpha

        # Build 80-bit initial state (big-endian encoding)
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

        # Burn-in: 160 clock cycles
        for _ in range(160):
            self._clock()

    @staticmethod
    def _to_bits(value, width):
        """Convert integer to big-endian bit list of given width."""
        return [int(b) for b in bin(value)[2:].zfill(width)]

    def _clock(self):
        """Clock the LFSR once, returning the new feedback bit."""
        new_bit = 0
        for tap in self.TAPS:
            new_bit ^= self.state[tap]
        self.state.pop(0)
        self.state.append(new_bit)
        return new_bit

    def next_bit(self):
        """
        Generate next output bit using self-shrinking mode.

        Consume raw LFSR bits in pairs (a, b):
          - If a = 1, output b
          - If a = 0, discard both and try next pair
        """
        while True:
            a = self._clock()
            b = self._clock()
            if a == 1:
                return b

    def random_field_element(self, n, p):
        """
        Sample a uniform random element from GF(p) using rejection sampling.

        Generates n-bit integers (MSB first) and rejects values >= p.
        """
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
    Check whether (R_F, R_P) satisfies all security inequalities for the
    Poseidon/Poseidon2 hash function with security level M bits.

    Implements the six constraints from:
      - Grassi et al., "POSEIDON" (eprint 2019/458), Section 5 + Appendix C
      - Kales & Zaverucha (eprint 2023/537) for the binomial bound

    Args:
        p: Field prime
        t: State width
        R_F, R_P: Number of full/partial rounds (before adding security margin)
        alpha: S-box degree
        M: Security level in bits (typically 128)
        n: Bit length of p

    Returns:
        True if all constraints are satisfied.
    """
    # Constraint 1: Statistical attack
    threshold = (floor(log(p, 2) - ((alpha - 1) / 2.0))) * (t + 1)
    R_F_1 = 6 if M <= threshold else 10

    # Constraint 2: Interpolation attack
    R_F_2 = 1 + ceil(log(2, alpha) * min(M, n)) + ceil(log(t, alpha)) - R_P

    # Constraint 3: Grobner basis attack variant 1
    R_F_3 = log(2, alpha) * min(M, log(p, 2)) - R_P

    # Constraint 4: Grobner basis attack variant 2
    R_F_4 = t - 1 + log(2, alpha) * min(M / float(t + 1), log(p, 2) / 2.0) - R_P

    # Constraint 5: Grobner basis attack variant 3
    R_F_5 = (t - 2 + (M / (2.0 * log(alpha, 2))) - R_P) / float(t - 1)

    R_F_max = max(ceil(R_F_1), ceil(R_F_2), ceil(R_F_3), ceil(R_F_4), ceil(R_F_5))

    # Constraint 6: Binomial attack (eprint 2023/537)
    r_temp = floor(t / 3.0)
    over = (R_F - 1) * t + R_P + r_temp + r_temp * (R_F / 2.0) + R_P + alpha
    under = r_temp * (R_F / 2.0) + R_P + alpha

    # Compute log2(C(over, under)) using Python's math.comb (3.8+)
    try:
        from math import comb

        binom_val = comb(int(over), int(under))
        if binom_val == 0:
            binom_log = 0
        else:
            binom_log = log2(binom_val)
    except (ValueError, OverflowError):
        binom_log = M + 1  # Treat overflow as satisfying the constraint

    cost_gb4 = ceil(2 * binom_log)

    return (R_F >= R_F_max) and (cost_gb4 >= M)


def compute_round_numbers(p, t, alpha, M=128):
    """
    Compute optimal (R_F, R_P) for given parameters with security margin.

    The security margin adds 2 to R_F and multiplies R_P by 1.075 (ceiling).

    Returns:
        (R_F, R_P) tuple
    """
    n = p.bit_length()

    best_R_F = 0
    best_R_P = 0
    min_cost = float("inf")
    max_cost_rf = 0

    # Brute-force search minimizing S-box cost = t * R_F + R_P
    for R_P_t in range(1, 500):
        for R_F_t in range(4, 100):
            if R_F_t % 2 != 0:
                continue
            if sat_inequalities(p, t, R_F_t, R_P_t, alpha, M, n):
                # Apply security margin
                R_F_m = R_F_t + 2
                R_P_m = int(ceil(R_P_t * 1.075))

                cost = t * R_F_m + R_P_m
                if (cost < min_cost) or (cost == min_cost and R_F_m < max_cost_rf):
                    best_R_P = R_P_m
                    best_R_F = R_F_m
                    min_cost = cost
                    max_cost_rf = best_R_F

    if best_R_F == 0:
        raise ValueError(
            f"No valid round numbers found for p={p}, t={t}, alpha={alpha}"
        )

    return (best_R_F, best_R_P)


# =============================================================================
# Round Constant Generation (Poseidon2)
# =============================================================================


def generate_round_constants_poseidon2(grain, p, n, t, R_F, R_P):
    """
    Generate Poseidon2 round constants from the Grain LFSR.

    Total raw constants: R_F * t + R_P
    (External rounds have t constants each; internal rounds have 1 constant each.)

    Returns:
        (external_initial, internal, external_final) where:
          external_initial: list of R_F/2 lists of t constants
          internal: list of R_P constants
          external_final: list of R_F/2 lists of t constants
    """
    R_f = R_F // 2
    num_constants = R_F * t + R_P

    # Generate all raw constants from Grain
    raw = []
    for _ in range(num_constants):
        raw.append(grain.random_field_element(n, p))

    # Split into external initial, internal, external final
    idx = 0
    external_initial = []
    for _ in range(R_f):
        external_initial.append(raw[idx : idx + t])
        idx += t

    internal = raw[idx : idx + R_P]
    idx += R_P

    external_final = []
    for _ in range(R_f):
        external_final.append(raw[idx : idx + t])
        idx += t

    assert idx == num_constants
    return external_initial, internal, external_final


# =============================================================================
# External Matrix (Deterministic)
# =============================================================================

# The fixed 4x4 M4 matrix used in Poseidon2
M4 = [
    [5, 7, 1, 3],
    [4, 6, 1, 1],
    [1, 3, 5, 7],
    [1, 1, 4, 6],
]


def generate_external_matrix(t, p):
    """
    Generate the Poseidon2 external (full) matrix M_E.

    For t = 4: M_E = M4
    For t divisible by 4 (t >= 8): block circulant with M4.
      M_E = circ(2*M4, M4, M4, ..., M4)
      i.e. diagonal 4x4 blocks get 2*M4, off-diagonal get M4.
    For t = 2: circ(2, 1)
    For t = 3: circ(2, 1, 1)
    """
    if t == 2:
        return [[2 % p, 1], [1, 2 % p]]
    elif t == 3:
        return [[2 % p, 1, 1], [1, 2 % p, 1], [1, 1, 2 % p]]
    elif t == 4:
        return [[x % p for x in row] for row in M4]
    elif t % 4 == 0:
        M = [[0] * t for _ in range(t)]
        num_blocks = t // 4
        for i in range(num_blocks):
            for j in range(num_blocks):
                factor = 2 if i == j else 1
                for r in range(4):
                    for c in range(4):
                        M[i * 4 + r][j * 4 + c] = (factor * M4[r][c]) % p
        return M
    else:
        raise ValueError(
            f"Unsupported width t={t} for external matrix (must be 2, 3, or divisible by 4)"
        )


# =============================================================================
# Internal Matrix (Grain-Generated Diagonal)
# =============================================================================


def check_minpoly_condition(M, t, p):
    """
    Check that for i = 1..2t, the minimal polynomial of M^i is irreducible
    of degree t over GF(p).

    Since an irreducible polynomial of degree t has no nontrivial divisors,
    if the characteristic polynomial is irreducible, then it equals the
    minimal polynomial. So we check: char_poly(M^i) is irreducible for all i.
    """
    M_pow = [row[:] for row in M]  # M^1
    for i in range(1, 2 * t + 1):
        cp = char_poly(M_pow, p)
        if not is_irreducible(cp, p):
            return False
        M_pow = mat_mul(M, M_pow, p)
    return True


def generate_internal_matrix(grain, t, n, p, verbose=False):
    """
    Generate the Poseidon2 internal matrix M_I = I + circ(0,1,...,1) + diag(v)
    where v is sampled from the Grain LFSR.

    The diagonal vector v is resampled until check_minpoly_condition passes.

    Returns:
        diag_minus_1: list of t values, the diagonal of (M_I - I)
    """
    attempt = 0
    while True:
        attempt += 1
        # Sample diagonal from Grain
        diag = [grain.random_field_element(n, p) for _ in range(t)]

        # Build M = circ(0,1,...,1) + diag(v)
        # circ(0,1,...,1)[i][j] = 1 if j != i, else 0
        # So M[i][j] = (1 if j != i else 0) + (diag[i] if i == j else 0)
        #            = diag[i] if i == j, else 1
        M = [[0] * t for _ in range(t)]
        for i in range(t):
            for j in range(t):
                if i == j:
                    M[i][j] = diag[i] % p
                else:
                    M[i][j] = 1

        if check_minpoly_condition(M, t, p):
            if verbose:
                print(f"  Internal matrix found after {attempt} attempt(s)")
            # diag_minus_1 = diagonal entries of (M - I)
            diag_minus_1 = [(diag[i] - 1) % p for i in range(t)]
            return diag_minus_1


# =============================================================================
# Poseidon2 Reference Permutation
# =============================================================================


def poseidon2_permutation(
    state,
    external_matrix,
    internal_matrix_diag_m1,
    external_initial,
    internal_constants,
    external_final,
    alpha,
    p,
    t,
):
    """
    Reference implementation of the Poseidon2 permutation.

    Args:
        state: list of t field elements (input)
        external_matrix: t x t matrix for external rounds
        internal_matrix_diag_m1: diagonal of (M_I - I), length t
        external_initial: R_F/2 lists of t round constants
        internal_constants: R_P scalar round constants
        external_final: R_F/2 lists of t round constants
        alpha: S-box degree
        p: field prime
        t: state width

    Returns:
        list of t field elements (output)
    """
    state = list(state)

    # Initial linear layer
    state = mat_vec_mul(external_matrix, state, p)

    # External rounds (initial)
    for rc in external_initial:
        # Add round constants
        for i in range(t):
            state[i] = (state[i] + rc[i]) % p
        # S-box on all elements
        for i in range(t):
            state[i] = pow(state[i], alpha, p)
        # External matrix
        state = mat_vec_mul(external_matrix, state, p)

    # Internal rounds
    for rc in internal_constants:
        # Add round constant to s0 only
        state[0] = (state[0] + rc) % p
        # S-box on s0 only
        state[0] = pow(state[0], alpha, p)
        # Internal matrix: M_I * state where M_I = I + circ(0,1,...,1) + diag(v-1)
        # M_I[i][j] = (v[i] if i==j else 1)
        # So M_I * x = sum(x) + (v[i]-1)*x[i] for each i
        total = sum(state) % p
        new_state = [0] * t
        for i in range(t):
            new_state[i] = (total + internal_matrix_diag_m1[i] * state[i]) % p
        state = new_state

    # External rounds (final)
    for rc in external_final:
        for i in range(t):
            state[i] = (state[i] + rc[i]) % p
        for i in range(t):
            state[i] = pow(state[i], alpha, p)
        state = mat_vec_mul(external_matrix, state, p)

    return state


# =============================================================================
# Output Formatting
# =============================================================================


def format_hex(value, n):
    """Format a field element as a hex string with appropriate width."""
    hex_width = (n + 3) // 4  # number of hex digits
    return f"0x{value:0{hex_width}x}"


def print_embedded_poseidon2_test_vector(field_name, width, n):
    """
    Print canonical input / expected output copied from Plonky3 Rust tests (or one-shot
    derivations documented in DEFAULT_POSEIDON2_TEST_VECTORS) for reviewer spot-checks.
    """
    key = (field_name, width)
    if key not in DEFAULT_POSEIDON2_TEST_VECTORS:
        print(
            f"\nNo embedded test vector for {field_name} width {width}.",
            file=sys.stderr,
        )
        return

    entry = DEFAULT_POSEIDON2_TEST_VECTORS[key]
    p = FIELDS[field_name]["prime"]

    print()
    print("Embedded default Poseidon2 permutation vector (Plonky3 canonical literals)")
    print(f"  Field: {field_name}  Width: {width}")
    print(f"  Rust permutation: {entry['permutation']}")
    for src in entry["rust_tests"]:
        print(f"  Source: {src}")
    print()

    inp = entry["input"]
    exp = entry["expected"]
    print("  Input (canonical, decimal):")
    print(f"    {inp}")
    print("  Input (hex):")
    print(f"    [{', '.join(format_hex(v, n) for v in inp)}]")
    print()
    print("  Expected output (canonical, decimal):")
    print(f"    {exp}")
    print("  Expected output (hex):")
    print(f"    [{', '.join(format_hex(v, n) for v in exp)}]")


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


def format_default_poseidon2(
    field_name, width, external_initial, internal, external_final,
    diag_m1, p, n, alpha, R_F, R_P, skip_matrix,
):
    """
    Format Poseidon2 constants as a clean, language-neutral summary.

    This is the default output: human-readable, no language-specific syntax.
    """
    R_f = R_F // 2
    sep = "─" * 72
    lines = []

    lines.append(sep)
    lines.append(f"  Poseidon2 Constants — {field_name} (width {width})")
    lines.append(sep)
    lines.append("")
    lines.append(f"  Field        {field_name}")
    lines.append(f"  Prime (p)    {p}")
    lines.append(f"  Bit length   {n}")
    lines.append(f"  S-box (α)    x^{alpha}")
    lines.append(f"  Width (t)    {width}")
    lines.append(f"  Full rounds  {R_F}  ({R_f} initial + {R_f} final)")
    lines.append(f"  Partial      {R_P}")
    lines.append(f"  Constants    {R_F * width + R_P}  ({R_F}×{width} + {R_P})")
    lines.append("")

    # External initial
    lines.append(sep)
    lines.append(f"  External Round Constants — Initial ({R_f} rounds × {width})")
    lines.append(sep)
    for i, rnd in enumerate(external_initial):
        lines.append("")
        lines.append(f"  round {i}:")
        lines.extend(_wrap_hex_row(rnd, n))

    lines.append("")

    # Internal
    lines.append(sep)
    lines.append(f"  Internal Round Constants ({R_P} scalars)")
    lines.append(sep)
    lines.append("")
    lines.extend(_wrap_hex_row(internal, n))

    lines.append("")

    # External final
    lines.append(sep)
    lines.append(f"  External Round Constants — Final ({R_f} rounds × {width})")
    lines.append(sep)
    for i, rnd in enumerate(external_final):
        lines.append("")
        lines.append(f"  round {R_f + R_P + i}:")
        lines.extend(_wrap_hex_row(rnd, n))

    # Internal matrix diagonal
    if not skip_matrix:
        lines.append("")
        lines.append(sep)
        lines.append(f"  Internal Matrix Diagonal − 1  (Grain-generated, {width} entries)")
        lines.append(f"  note: production builds use hand-optimized diagonals")
        lines.append(sep)
        lines.append("")
        lines.extend(_wrap_hex_row(diag_m1, n))

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)



def format_json_poseidon2(
    field_name, width, external_initial, internal, external_final,
    diag_m1, p, n, alpha, R_F, R_P, skip_matrix,
):
    """Format Poseidon2 constants as JSON."""
    data = {
        "field": field_name,
        "prime": str(p),
        "width": width,
        "alpha": alpha,
        "R_F": R_F,
        "R_P": R_P,
        "external_initial": [
            [format_hex(v, n) for v in rnd] for rnd in external_initial
        ],
        "internal": [format_hex(v, n) for v in internal],
        "external_final": [[format_hex(v, n) for v in rnd] for rnd in external_final],
    }
    if not skip_matrix:
        data["matrix_diag_minus_1_grain"] = [format_hex(v, n) for v in diag_m1]
    return json.dumps(data, indent=2)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate Poseidon2 round constants for various prime fields.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  python poseidon2/generate_constants.py --field babybear --width 16\n"
        "  python poseidon2/generate_constants.py --field goldilocks --width 8 --format json\n"
        "  python poseidon2/generate_constants.py --field koalabear --width 24 -v\n"
        "  python poseidon2/generate_constants.py --field babybear --width 16 --skip-matrix --test-vector\n",
    )
    parser.add_argument(
        "--field",
        required=True,
        choices=list(FIELDS.keys()),
        help="Target field",
    )
    parser.add_argument(
        "--width",
        required=True,
        type=int,
        help="State width (t)",
    )
    parser.add_argument(
        "--format",
        default="default",
        choices=["default", "json"],
        help="Output format (default: human-readable summary)",
    )
    parser.add_argument(
        "--security-level",
        default=128,
        type=int,
        help="Security level in bits (default: 128)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose progress information",
    )
    parser.add_argument(
        "--skip-matrix",
        action="store_true",
        help="Skip internal matrix generation (faster; only generates round constants)",
    )
    parser.add_argument(
        "--test-vector",
        action="store_true",
        help=(
            "Print embedded canonical input/expected output for the default Poseidon2 permutation "
            "(from Rust tests or documented one-shot derivations; not from the Python reference)"
        ),
    )
    args = parser.parse_args()

    field_info = FIELDS[args.field]
    p = field_info["prime"]
    t = args.width

    if t not in field_info["valid_widths"]:
        print(
            f"Error: width {t} not valid for {args.field}. "
            f"Valid widths: {field_info['valid_widths']}",
            file=sys.stderr,
        )
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
        print(f"Total raw constants: R_F*t + R_P = {R_F * t + R_P}")
        print()

    # --- Initialize Grain LFSR ---
    grain = GrainLFSR(n, t, R_F, R_P)

    # --- Generate round constants ---
    if args.verbose:
        print("Generating round constants...", flush=True)
    external_initial, internal, external_final = generate_round_constants_poseidon2(
        grain, p, n, t, R_F, R_P
    )

    # --- Generate internal matrix diagonal ---
    if not args.skip_matrix:
        if args.verbose:
            print("Generating internal matrix (this may take a moment)...", flush=True)
        diag_m1 = generate_internal_matrix(grain, t, n, p, verbose=args.verbose)
    else:
        diag_m1 = [0] * t
        if args.verbose:
            print("Skipping internal matrix generation")

    if args.verbose:
        print()

    # --- Output ---
    fmt_args = (
        args.field, t, external_initial, internal, external_final,
        diag_m1, p, n, alpha, R_F, R_P, args.skip_matrix,
    )
    if args.format == "default":
        print(format_default_poseidon2(*fmt_args))
    elif args.format == "json":
        print(format_json_poseidon2(*fmt_args))

    # --- Test vector (embedded Plonky3 literals; independent of --skip-matrix) ---
    if args.test_vector:
        print_embedded_poseidon2_test_vector(args.field, t, n)


if __name__ == "__main__":
    main()
