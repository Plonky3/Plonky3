#!/usr/bin/env python3

# Debug Montgomery parameters for Goldilocks field
P = 2**64 - 2**32 + 1  # 0xffffffff00000001

print(f"P = {P}")
print(f"P = {hex(P)}")

# R = 2^64 mod P
# Since 2^64 = P + 2^32 - 1, we have 2^64 ≡ 2^32 - 1 (mod P)
R = (2**64) % P
print(f"R = 2^64 mod P = {R}")
print(f"R = {hex(R)}")

# R^2 mod P
R2 = (R * R) % P
print(f"R^2 mod P = {R2}")
print(f"R^2 = {hex(R2)}")

# Extended GCD to find modular inverse
def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

# Find P^{-1} mod 2^64
# We want to find x such that P * x ≡ 1 (mod 2^64)
gcd, x, y = extended_gcd(P, 2**64)
print(f"gcd(P, 2^64) = {gcd}")

if gcd == 1:
    # P^{-1} mod 2^64
    P_inv = x % (2**64)
    print(f"P^-1 mod 2^64 = {P_inv}")
    print(f"P^-1 = {hex(P_inv)}")
    
    # -P^{-1} mod 2^64
    neg_P_inv = (-P_inv) % (2**64)
    print(f"-P^-1 mod 2^64 = {neg_P_inv}")
    print(f"-P^-1 = {hex(neg_P_inv)}")
    
    # Verify: P * P^{-1} ≡ 1 (mod 2^64)
    check = (P * P_inv) % (2**64)
    print(f"Check: P * P^-1 mod 2^64 = {check}")

# Montgomery forms
print("\nMontgomery forms:")
print(f"0 in Montgomery: {0}")
print(f"1 in Montgomery: {R} = {hex(R)}")
print(f"2 in Montgomery: {(2 * R) % P} = {hex((2 * R) % P)}")
print(f"-1 in Montgomery: {((P-1) * R) % P} = {hex(((P-1) * R) % P)}")

# Test conversion
def to_montgomery(x):
    return (x * R2) % P

def from_montgomery(x):
    # Montgomery reduction
    # This is a simplified version - the actual implementation needs proper handling
    return (x * pow(R, P-2, P)) % P

print(f"\nTest: 1 -> Montgomery -> back:")
one_mont = to_montgomery(1)
print(f"1 in Montgomery: {one_mont} = {hex(one_mont)}")
one_back = from_montgomery(one_mont) 
print(f"Back to normal: {one_back}")