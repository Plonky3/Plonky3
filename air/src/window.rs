use p3_matrix::dense::RowMajorMatrixView;

/// Read access to a pair of trace rows (typically current and next).
///
/// Implementors expose two flat slices that constraint evaluators use
/// to express algebraic relations between rows.
pub trait WindowAccess<T> {
    /// Full slice of the current row.
    fn current_slice(&self) -> &[T];

    /// Full slice of the next row.
    fn next_slice(&self) -> &[T];

    /// Single element from the current row by index.
    ///
    /// Returns `None` if `i` is out of bounds.
    #[inline]
    fn current(&self, i: usize) -> Option<T>
    where
        T: Clone,
    {
        self.current_slice().get(i).cloned()
    }

    /// Single element from the next row by index.
    ///
    /// Returns `None` if `i` is out of bounds.
    #[inline]
    fn next(&self, i: usize) -> Option<T>
    where
        T: Clone,
    {
        self.next_slice().get(i).cloned()
    }
}

/// A lightweight two-row window into a trace matrix.
///
/// Stores two `&[T]` slices — one for the current row and one for
/// the next — without carrying any matrix metadata.  This is cheaper
/// than a full `ViewPair` and is the concrete type used by most
/// [`AirBuilder`] implementations for `type MainWindow` / `type PreprocessedWindow`.
#[derive(Debug, Clone, Copy)]
pub struct RowWindow<'a, T> {
    /// The current row.
    current: &'a [T],
    /// The next row.
    next: &'a [T],
}

impl<'a, T> RowWindow<'a, T> {
    /// Create a window from a [`RowMajorMatrixView`] that has exactly
    /// two rows. The first row becomes `current`, the second `next`.
    ///
    /// # Panics
    ///
    /// Panics if the view does not contain exactly `2 * width` elements.
    #[inline]
    pub fn from_view(view: &RowMajorMatrixView<'a, T>) -> Self {
        let width = view.width;
        assert_eq!(
            view.values.len(),
            2 * width,
            "RowWindow::from_view: expected 2 rows (2*{width} elements), got {}",
            view.values.len()
        );
        let (current, next) = view.values.split_at(width);
        Self { current, next }
    }

    /// Create a window from two separate row slices.
    ///
    /// The caller is responsible for providing slices that represent
    /// the intended (current, next) pair.
    ///
    /// # Panics
    ///
    /// Panics (in debug builds) if the slices have different lengths.
    #[inline]
    pub fn from_two_rows(current: &'a [T], next: &'a [T]) -> Self {
        debug_assert_eq!(
            current.len(),
            next.len(),
            "RowWindow::from_two_rows: row lengths differ ({} vs {})",
            current.len(),
            next.len()
        );
        Self { current, next }
    }
}

impl<T> WindowAccess<T> for RowWindow<'_, T> {
    #[inline]
    fn current_slice(&self) -> &[T] {
        self.current
    }

    #[inline]
    fn next_slice(&self) -> &[T] {
        self.next
    }
}
