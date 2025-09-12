#!/usr/bin/env python3

# Debug Montgomery multiplication
P = 0xffffffff00000001  # Goldilocks prime
MONTY_INV = 0xfffffffeffffffff  # -P^{-1} mod 2^64
R = 0xffffffff  # R = 2^64 mod P
R2 = 0xfffffffe00000001  # R^2 mod P

def montgomery_reduce(a):
    """Montgomery reduction: compute (a * R^{-1}) mod P"""
    a_lo = a & 0xffffffffffffffff
    m = (a_lo * MONTY_INV) & 0xffffffffffffffff
    mp = m * P
    t = (a + mp) >> 64
    t = t & 0xffffffffffffffff
    if t >= P:
        t -= P
    return t

def to_montgomery(x):
    """Convert x to Montgomery form"""
    return montgomery_reduce(x * R2)

def montgomery_mul(a, b):
    """Multiply two Montgomery form numbers"""
    return montgomery_reduce(a * b)

# Test the basic constants
print(f"P = {hex(P)}")
print(f"R = {hex(R)}")
print(f"R2 = {hex(R2)}")
print(f"MONTY_INV = {hex(MONTY_INV)}")

# Verify that Montgomery reduction of R2 gives 1
print(f"\nMontgomery reduction of R2: {hex(montgomery_reduce(R2))}")
print(f"Should be 1: {montgomery_reduce(R2) == 1}")

# Test 1 in Montgomery form
one_mont = to_montgomery(1)
print(f"\n1 in Montgomery: {hex(one_mont)}")
print(f"Should be R: {hex(R)}")
print(f"Correct: {one_mont == R}")

# Test multiplication: 1 * 1 should give 1 (in Montgomery form)
mult_result = montgomery_mul(one_mont, one_mont)
print(f"\n1_mont * 1_mont = {hex(mult_result)}")
print(f"Should be R: {hex(R)}")
print(f"Correct: {mult_result == R}")

# Test with actual values
a = 123456
a_mont = to_montgomery(a)
print(f"\n{a} in Montgomery: {hex(a_mont)}")

# a * 1 should equal a
a_times_one = montgomery_mul(a_mont, one_mont)
print(f"a * 1_mont = {hex(a_times_one)}")
print(f"Should equal a_mont: {a_times_one == a_mont}")