# Poseidon2 constants for BabyBear (p = 2013265921)
# Usage: sage poseidon2_babybear.sage
# Change t below for different widths (16 for compression, 24 for sponge)
from sage.rings.polynomial.polynomial_gf2x import GF2X_BuildIrred_list
from math import *
import itertools

###########################################################################
p = 2013265921 # BabyBear

n = len(p.bits()) # bit
t = 16 # BabyBear (t = 24 for sponge, t = 16 for compression)

FIELD = 1
SBOX = 0
FIELD_SIZE = n
NUM_CELLS = t

def get_alpha(p):
    for alpha in range(3, p):
        if gcd(alpha, p-1) == 1:
            break
    return alpha

alpha = get_alpha(p)

def get_sbox_cost(R_F, R_P, N, t):
    return int(t * R_F + R_P)

def get_size_cost(R_F, R_P, N, t):
    n = ceil(float(N) / t)
    return int((N * R_F) + (n * R_P))

def poseidon_calc_final_numbers_fixed(p, t, alpha, M, security_margin):
    n = ceil(log(p, 2))
    N = int(n * t)
    cost_function = get_sbox_cost
    ret_list = []
    (R_F, R_P) = find_FD_round_numbers(p, t, alpha, M, cost_function, security_margin)
    min_sbox_cost = cost_function(R_F, R_P, N, t)
    ret_list.append(R_F)
    ret_list.append(R_P)
    ret_list.append(min_sbox_cost)
    min_size_cost = get_size_cost(R_F, R_P, N, t)
    ret_list.append(min_size_cost)
    return ret_list

def find_FD_round_numbers(p, t, alpha, M, cost_function, security_margin):
    n = ceil(log(p, 2))
    N = int(n * t)
    sat_inequiv = sat_inequiv_alpha
    R_P = 0
    R_F = 0
    min_cost = float("inf")
    max_cost_rf = 0
    for R_P_t in range(1, 500):
        for R_F_t in range(4, 100):
            if R_F_t % 2 == 0:
                if (sat_inequiv(p, t, R_F_t, R_P_t, alpha, M) == True):
                    if security_margin == True:
                        R_F_t += 2
                        R_P_t = int(ceil(float(R_P_t) * 1.075))
                    cost = cost_function(R_F_t, R_P_t, N, t)
                    if (cost < min_cost) or ((cost == min_cost) and (R_F_t < max_cost_rf)):
                        R_P = ceil(R_P_t)
                        R_F = ceil(R_F_t)
                        min_cost = cost
                        max_cost_rf = R_F
    return (int(R_F), int(R_P))

def sat_inequiv_alpha(p, t, R_F, R_P, alpha, M):
    N = int(FIELD_SIZE * NUM_CELLS)
    if alpha > 0:
        R_F_1 = 6 if M <= ((floor(log(p, 2) - ((alpha-1)/2.0))) * (t + 1)) else 10
        R_F_2 = 1 + ceil(log(2, alpha) * min(M, FIELD_SIZE)) + ceil(log(t, alpha)) - R_P
        R_F_3 = (log(2, alpha) * min(M, log(p, 2))) - R_P
        R_F_4 = t - 1 + log(2, alpha) * min(M / float(t + 1), log(p, 2) / float(2)) - R_P
        R_F_5 = (t - 2 + (M / float(2 * log(alpha, 2))) - R_P) / float(t - 1)
        R_F_max = max(ceil(R_F_1), ceil(R_F_2), ceil(R_F_3), ceil(R_F_4), ceil(R_F_5))
        r_temp = floor(t / 3.0)
        over = (R_F - 1) * t + R_P + r_temp + r_temp * (R_F / 2.0) + R_P + alpha
        under = r_temp * (R_F / 2.0) + R_P + alpha
        binom_log = log(binomial(over, under), 2)
        if binom_log == inf:
            binom_log = M + 1
        cost_gb4 = ceil(2 * binom_log)
        return ((R_F >= R_F_max) and (cost_gb4 >= M))
    else:
        print("Invalid value for alpha!")
        exit(1)

R_F_FIXED, R_P_FIXED, _, _ = poseidon_calc_final_numbers_fixed(p, t, alpha, 128, True)
print("+++ R_F = {0}, R_P = {1} +++".format(R_F_FIXED, R_P_FIXED))

###########################################################################

INIT_SEQUENCE = []
PRIME_NUMBER = p
F = GF(PRIME_NUMBER)

def grain_sr_generator():
    bit_sequence = INIT_SEQUENCE
    for _ in range(0, 160):
        new_bit = bit_sequence[62] ^^ bit_sequence[51] ^^ bit_sequence[38] ^^ bit_sequence[23] ^^ bit_sequence[13] ^^ bit_sequence[0]
        bit_sequence.pop(0)
        bit_sequence.append(new_bit)
    while True:
        new_bit = bit_sequence[62] ^^ bit_sequence[51] ^^ bit_sequence[38] ^^ bit_sequence[23] ^^ bit_sequence[13] ^^ bit_sequence[0]
        bit_sequence.pop(0)
        bit_sequence.append(new_bit)
        while new_bit == 0:
            new_bit = bit_sequence[62] ^^ bit_sequence[51] ^^ bit_sequence[38] ^^ bit_sequence[23] ^^ bit_sequence[13] ^^ bit_sequence[0]
            bit_sequence.pop(0)
            bit_sequence.append(new_bit)
            new_bit = bit_sequence[62] ^^ bit_sequence[51] ^^ bit_sequence[38] ^^ bit_sequence[23] ^^ bit_sequence[13] ^^ bit_sequence[0]
            bit_sequence.pop(0)
            bit_sequence.append(new_bit)
        new_bit = bit_sequence[62] ^^ bit_sequence[51] ^^ bit_sequence[38] ^^ bit_sequence[23] ^^ bit_sequence[13] ^^ bit_sequence[0]
        bit_sequence.pop(0)
        bit_sequence.append(new_bit)
        yield new_bit
grain_gen = grain_sr_generator()

def grain_random_bits(num_bits):
    random_bits = [next(grain_gen) for i in range(0, num_bits)]
    random_int = int("".join(str(i) for i in random_bits), 2)
    return random_int

def init_generator(field, sbox, n, t, R_F, R_P):
    bit_list_field = [_ for _ in (bin(FIELD)[2:].zfill(2))]
    bit_list_sbox = [_ for _ in (bin(SBOX)[2:].zfill(4))]
    bit_list_n = [_ for _ in (bin(FIELD_SIZE)[2:].zfill(12))]
    bit_list_t = [_ for _ in (bin(NUM_CELLS)[2:].zfill(12))]
    bit_list_R_F = [_ for _ in (bin(R_F)[2:].zfill(10))]
    bit_list_R_P = [_ for _ in (bin(R_P)[2:].zfill(10))]
    bit_list_1 = [1] * 30
    global INIT_SEQUENCE
    INIT_SEQUENCE = bit_list_field + bit_list_sbox + bit_list_n + bit_list_t + bit_list_R_F + bit_list_R_P + bit_list_1
    INIT_SEQUENCE = [int(_) for _ in INIT_SEQUENCE]

def generate_constants(field, n, t, R_F, R_P, prime_number):
    round_constants = []
    num_constants = (R_F * t) + R_P # Poseidon2
    if field == 0:
        for i in range(0, num_constants):
            random_int = grain_random_bits(n)
            round_constants.append(random_int)
    elif field == 1:
        for i in range(0, num_constants):
            random_int = grain_random_bits(n)
            while random_int >= prime_number:
                random_int = grain_random_bits(n)
            round_constants.append(random_int)
            # Add (t-1) zeroes for Poseidon2 if partial round
            if i >= ((R_F/2) * t) and i < (((R_F/2) * t) + R_P):
                round_constants.extend([0] * (t-1))
    return round_constants

def create_mds_p(n, t):
    M = matrix(F, t, t)
    while True:
        flag = True
        rand_list = [F(grain_random_bits(n)) for _ in range(0, 2*t)]
        while len(rand_list) != len(set(rand_list)):
            rand_list = [F(grain_random_bits(n)) for _ in range(0, 2*t)]
        xs = rand_list[:t]
        ys = rand_list[t:]
        for i in range(0, t):
            for j in range(0, t):
                if (flag == False) or ((xs[i] + ys[j]) == 0):
                    flag = False
                else:
                    entry = (xs[i] + ys[j])^(-1)
                    M[i, j] = entry
        if flag == False:
            continue
        return M

def check_minpoly_condition(M, NUM_CELLS):
    max_period = 2*NUM_CELLS
    all_fulfilled = True
    M_temp = M
    for i in range(1, max_period + 1):
        if not ((M_temp.minimal_polynomial().degree() == NUM_CELLS) and (M_temp.minimal_polynomial().is_irreducible() == True)):
            all_fulfilled = False
            break
        M_temp = M * M_temp
    return all_fulfilled

def generate_matrix_full(NUM_CELLS):
    M = None
    if t == 2:
        M = matrix.circulant(vector([F(2), F(1)]))
    elif t == 3:
        M = matrix.circulant(vector([F(2), F(1), F(1)]))
    elif t == 4:
        M = matrix(F, [[F(5), F(7), F(1), F(3)], [F(4), F(6), F(1), F(1)], [F(1), F(3), F(5), F(7)], [F(1), F(1), F(4), F(6)]])
    elif (t % 4) == 0:
        M = matrix(F, t, t)
        M_small = matrix(F, [[F(5), F(7), F(1), F(3)], [F(4), F(6), F(1), F(1)], [F(1), F(3), F(5), F(7)], [F(1), F(1), F(4), F(6)]])
        small_num = t // 4
        for i in range(0, small_num):
            for j in range(0, small_num):
                if i == j:
                    M[i*4:(i+1)*4,j*4:(j+1)*4] = 2* M_small
                else:
                    M[i*4:(i+1)*4,j*4:(j+1)*4] = M_small
    else:
        print("Error: No matrix for these parameters.")
        exit()
    return M

def generate_matrix_partial(FIELD, FIELD_SIZE, NUM_CELLS):
    entry_max_bit_size = FIELD_SIZE
    if FIELD == 0:
        print("Matrix generation not implemented for GF(2^n).")
        exit(1)
    elif FIELD == 1:
        M = None
        if t == 2:
            M = matrix(F, [[F(2), F(1)], [F(1), F(3)]])
        elif t == 3:
            M = matrix(F, [[F(2), F(1), F(1)], [F(1), F(2), F(1)], [F(1), F(1), F(3)]])
        else:
            M_circulant = matrix.circulant(vector([F(0)] + [F(1) for _ in range(0, NUM_CELLS - 1)]))
            M_diagonal = matrix.diagonal([F(grain_random_bits(entry_max_bit_size)) for _ in range(0, NUM_CELLS)])
            M = M_circulant + M_diagonal
            while check_minpoly_condition(M, NUM_CELLS) == False:
                M_diagonal = matrix.diagonal([F(grain_random_bits(entry_max_bit_size)) for _ in range(0, NUM_CELLS)])
                M = M_circulant + M_diagonal
        return M

def matrix_partial_m_1(matrix_partial, NUM_CELLS):
    M_circulant = matrix.identity(F, NUM_CELLS)
    return matrix_partial - M_circulant

def poseidon2(input_words, matrix_full, matrix_partial, round_constants):
    R_f = int(R_F_FIXED / 2)
    round_constants_counter = 0
    state_words = list(input_words)
    state_words = list(matrix_full * vector(state_words))
    for r in range(0, R_f):
        for i in range(0, t):
            state_words[i] = state_words[i] + round_constants[round_constants_counter]
            round_constants_counter += 1
        for i in range(0, t):
            state_words[i] = (state_words[i])^alpha
        state_words = list(matrix_full * vector(state_words))
    for r in range(0, R_P_FIXED):
        for i in range(0, t):
            state_words[i] = state_words[i] + round_constants[round_constants_counter]
            round_constants_counter += 1
        state_words[0] = (state_words[0])^alpha
        state_words = list(matrix_partial * vector(state_words))
    for r in range(0, R_f):
        for i in range(0, t):
            state_words[i] = state_words[i] + round_constants[round_constants_counter]
            round_constants_counter += 1
        for i in range(0, t):
            state_words[i] = (state_words[i])^alpha
        state_words = list(matrix_full * vector(state_words))
    return state_words

# Init
init_generator(FIELD, SBOX, FIELD_SIZE, NUM_CELLS, R_F_FIXED, R_P_FIXED)

# Round constants
round_constants = generate_constants(FIELD, FIELD_SIZE, NUM_CELLS, R_F_FIXED, R_P_FIXED, PRIME_NUMBER)

# Matrices
MATRIX_FULL = generate_matrix_full(NUM_CELLS)
MATRIX_PARTIAL = generate_matrix_partial(FIELD, FIELD_SIZE, NUM_CELLS)
MATRIX_PARTIAL_DIAGONAL_M_1 = [matrix_partial_m_1(MATRIX_PARTIAL, NUM_CELLS)[i,i] for i in range(0, NUM_CELLS)]

# Output
hex_length = int(ceil(float(n) / 4)) + 2

print("Partial matrix diagonal - 1:")
print(["{0:#0{1}x}".format(int(val), hex_length) for val in MATRIX_PARTIAL_DIAGONAL_M_1])
print()

print("Round constants (per round, {} elements each):".format(t))
print("Number of round constants:", len(round_constants))
for r in range(0, R_F_FIXED + R_P_FIXED):
    rc_round = round_constants[r*t:(r+1)*t]
    print("Round {}: {}".format(r, ["{0:#0{1}x}".format(entry, hex_length) for entry in rc_round]))
print()

state_in = vector([F(i) for i in range(t)])
state_out = poseidon2(state_in, MATRIX_FULL, MATRIX_PARTIAL, round_constants)
print("Input:  {}".format(["{0:#0{1}x}".format(int(val), hex_length) for val in state_in]))
print("Output: {}".format(["{0:#0{1}x}".format(int(val), hex_length) for val in state_out]))
