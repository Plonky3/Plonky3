#!/usr/bin/env python3

# Debug the inverse test
P = 2**64 - 2**32 + 1  # Goldilocks prime
R = (2**64) % P  # R = 0xffffffff

def to_montgomery(x):
    R2 = 0xfffffffe00000001
    return (x * R2) % P

def from_montgomery(x):
    # Simplified Montgomery reduction
    return (x * pow(R, P-2, P)) % P

# Test value from the failing test
a = 123456
print(f"Testing a = {a}")

# Convert to Montgomery form
a_mont = to_montgomery(a)
print(f"a in Montgomery: {a_mont} = {hex(a_mont)}")

# Calculate inverse using Fermat's little theorem
exp = P - 2
a_inv_mont = pow(a_mont, exp, P)  # This should be the Montgomery form of a^{-1}
print(f"a^-1 in Montgomery: {a_inv_mont} = {hex(a_inv_mont)}")

# Convert back to check
a_inv = from_montgomery(a_inv_mont)
print(f"a^-1 in normal form: {a_inv}")

# Check: a * a^{-1} should equal 1
check = (a * a_inv) % P
print(f"a * a^-1 mod P = {check}")

# In Montgomery form, multiplication should give Montgomery form of 1
mont_product = (a_mont * a_inv_mont) % P
print(f"a_mont * a_inv_mont mod P = {mont_product} = {hex(mont_product)}")
print(f"Expected (R mod P) = {R} = {hex(R)}")

# The issue might be that we're doing exponentiation on Montgomery form directly
# Let's try the correct approach: convert to normal, invert, convert back
a_normal = from_montgomery(a_mont)
print(f"a_mont -> normal: {a_normal}")
a_inv_normal = pow(a_normal, P-2, P)
print(f"a^-1 normal: {a_inv_normal}")
a_inv_mont_correct = to_montgomery(a_inv_normal)
print(f"a^-1 -> Montgomery: {a_inv_mont_correct} = {hex(a_inv_mont_correct)}")

# Check
mont_product_correct = (a_mont * a_inv_mont_correct) % P
print(f"Correct product: {mont_product_correct} = {hex(mont_product_correct)}")