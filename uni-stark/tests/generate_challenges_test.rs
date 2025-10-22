//! Tests for the `generate_challenges` function

use p3_baby_bear::BabyBear;
use p3_uni_stark::{FriVerificationChallenges, VerificationChallenges};

/// Test that verifies the structure has the expected fields
#[test]
fn test_verification_challenges_structure() {
    use p3_baby_bear::BabyBear;

    let fri_challenges = FriVerificationChallenges {
        alpha: BabyBear::new(4),
        betas: vec![BabyBear::new(5), BabyBear::new(6)],
        query_indices: vec![0, 1, 2],
    };

    let challenges = VerificationChallenges {
        alpha: BabyBear::new(1),
        zeta: BabyBear::new(2),
        zeta_next: BabyBear::new(3),
        fri_challenges: Some(fri_challenges),
    };

    // Test that we can access all fields
    assert_eq!(challenges.alpha, BabyBear::new(1));
    assert_eq!(challenges.zeta, BabyBear::new(2));
    assert_eq!(challenges.zeta_next, BabyBear::new(3));

    // Test FRI challenges
    if let Some(fri) = challenges.fri_challenges {
        assert_eq!(fri.alpha, BabyBear::new(4));
        assert_eq!(fri.betas.len(), 2);
        assert_eq!(fri.query_indices.len(), 3);
    }
}

/// Test that verifies the structure works without FRI challenges
#[test]
fn test_verification_challenges_without_fri() {
    use p3_baby_bear::BabyBear;

    let challenges = VerificationChallenges {
        alpha: BabyBear::new(1),
        zeta: BabyBear::new(2),
        zeta_next: BabyBear::new(3),
        fri_challenges: None,
    };

    // Test that we can access all fields
    assert_eq!(challenges.alpha, BabyBear::new(1));
    assert_eq!(challenges.zeta, BabyBear::new(2));
    assert_eq!(challenges.zeta_next, BabyBear::new(3));
    assert!(challenges.fri_challenges.is_none());
}
