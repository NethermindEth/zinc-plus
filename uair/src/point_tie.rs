//! Point ties: the cells a statement fixes.
//!
//! A read at a position the statement already knows is geometry, not a
//! query. Where a pointer query spells its position out of committed bit
//! columns because the prover chooses it, a tie names the cell in the
//! signature, so the verifier evaluates the cell's `eq` itself and
//! nothing about the position enters the proof.

/// What a tied cell's value is fixed to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PointTieTarget {
    /// A public value the cell must hold.
    Value(u64),
    /// Another integer column, which the tie makes constant and equal to
    /// the tied cell. The cell's value is private and stays private: the
    /// broadcast column carries it at every row, so ordinary uniform
    /// constraints can read it wherever they run.
    Broadcast(usize),
}

/// One cell of the trace the statement fixes: row `row` of column
/// `column`, both flat-indexed as the trace is (`binary_poly ||
/// arbitrary_poly || int`) and both public structure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointTie {
    pub column: usize,
    pub row: usize,
    pub target: PointTieTarget,
}

impl PointTie {
    /// The cell holds `value`.
    pub fn pin(column: usize, row: usize, value: u64) -> Self {
        Self {
            column,
            row,
            target: PointTieTarget::Value(value),
        }
    }

    /// The cell's value fills `into`, every row of it.
    pub fn broadcast(column: usize, row: usize, into: usize) -> Self {
        Self {
            column,
            row,
            target: PointTieTarget::Broadcast(into),
        }
    }

    /// How many terms the tie adds to the constraint composition: a pin
    /// says where a value sits, a broadcast also says its column is
    /// constant.
    pub fn num_terms(&self) -> usize {
        match self.target {
            PointTieTarget::Value(_) => 1,
            PointTieTarget::Broadcast(_) => 2,
        }
    }

    /// The columns the tie reads, the tied cell's first.
    pub fn columns(&self) -> Vec<usize> {
        match self.target {
            PointTieTarget::Value(_) => vec![self.column],
            PointTieTarget::Broadcast(into) => vec![self.column, into],
        }
    }
}
