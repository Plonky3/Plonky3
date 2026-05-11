/// Manages public input declarations and tracking.
#[derive(Debug, Clone, Default)]
pub struct PublicInputTracker {
    /// The number of public inputs declared
    count: usize,
}

impl PublicInputTracker {
    /// Creates a new public input tracker.
    pub const fn new() -> Self {
        Self { count: 0 }
    }

    /// Allocates the next public input position.
    ///
    /// Returns the position of the newly allocated public input.
    pub const fn alloc(&mut self) -> usize {
        let pos = self.count;
        self.count += 1;
        pos
    }

    /// Returns the total count of public inputs.
    pub const fn count(&self) -> usize {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_public_input_tracker_basic() {
        let mut tracker = PublicInputTracker::new();
        assert_eq!(tracker.count(), 0);

        let pos0 = tracker.alloc();
        assert_eq!(pos0, 0);
        assert_eq!(tracker.count(), 1);

        let pos1 = tracker.alloc();
        assert_eq!(pos1, 1);
        assert_eq!(tracker.count(), 2);

        let pos2 = tracker.alloc();
        assert_eq!(pos2, 2);
        assert_eq!(tracker.count(), 3);
    }

    #[test]
    fn test_public_input_tracker_default() {
        let tracker = PublicInputTracker::default();
        assert_eq!(tracker.count(), 0);
    }
}
