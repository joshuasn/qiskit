// This code is a Qiskit project.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! The symbolic angles absorbed into collectors, and the symbols a caller has to bind.
//!
//! A collector folds its absorbed gates into the angles it synthesizes, so an absorbed gate's angle is
//! an input to sampling rather than something the template executes. Bound angles need nothing from
//! this table — they travel inline on
//! [`AbsorbedGate`](crate::virtual_flow_graph::AbsorbedGate) as
//! [`AbsorbedParam::Bound`](crate::virtual_flow_graph::AbsorbedParam::Bound). What a table is for is
//! the question no individual gate can answer: **what must the caller supply before drawing.**
//! [`ParameterTable::free`] is that list, the parameter analogue of
//! [`DistributionTable::draw_counts`](crate::distributions::DistributionTable) sizing the sample
//! arrays.
//!
//! Interning mirrors [`DistributionTable`](crate::distributions::DistributionTable), which is the
//! other side object of the same shape — but not the same lifetime, which is why they are two
//! objects. The distribution table is an *input* to lowering, produced by the build pass; this one is
//! an *output*, because parameters are minted during lowering and deliberately nowhere earlier.
//!
//! The table is GIL-free: [`ParameterExpression`] is Rust-native (a `SymbolExpr` plus a name map), so
//! nothing here holds a Python object and the sampling graph stays plain data.

use std::sync::Arc;

use hashbrown::HashMap;
use pyo3::prelude::*;

use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
use qiskit_circuit::parameter::symbol_expr::Symbol;

/// A key into a [`ParameterTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParamKey(pub u32);

impl ParamKey {
    /// The index this key refers to.
    pub fn index(&self) -> usize {
        self.0 as usize
    }
}

/// The order two symbols are listed in, and the identity they are deduplicated by.
///
/// [`Symbol::fullname`] rather than [`Symbol::name`] because a `ParameterVector` element's `name` is
/// the bare vector name, shared by every element — it is `fullname` that a caller binds by. The uuid
/// is the tiebreak, so two symbols that share a name but differ in identity both appear rather than
/// one silently standing in for the other. It is taken as a `u128` so that samplex needs no `uuid`
/// dependency of its own.
fn ordering(symbol: &Symbol) -> (String, u128) {
    (symbol.fullname().into_owned(), symbol.uuid().as_u128())
}

/// The symbolic angles absorbed into collectors, and the symbols the caller must bind.
///
/// Every entry has at least one free symbol: a fully bound expression is folded to
/// [`AbsorbedParam::Bound`](crate::virtual_flow_graph::AbsorbedParam::Bound) as it is read and never
/// reaches here. So `free` being empty means there is nothing to bind, not that nothing was checked.
///
/// This table owns *only* the user's own parameters. The `p0000…` angles lowering mints for the synth
/// templates are a separate space, addressed by index into the template's own parameter vector, so a
/// user parameter that happens to be named `p0000` is not a collision here — but a consumer must not
/// merge the two lists by name, because it would become one there.
#[pyclass(module = "qiskit._accelerate.samplex", skip_from_py_object)]
#[derive(Debug, Clone, Default)]
pub struct ParameterTable {
    entries: Vec<Arc<ParameterExpression>>,
    lookup: HashMap<Arc<ParameterExpression>, ParamKey>,
    /// Every symbol appearing in any entry, deduplicated, ordered by [`ordering`].
    free: Vec<Symbol>,
}

impl ParameterTable {
    /// Construct an empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `expr`, returning the key for it. Returns the existing key if it is already present.
    pub fn intern(&mut self, expr: Arc<ParameterExpression>) -> ParamKey {
        if let Some(key) = self.lookup.get(&expr) {
            return *key;
        }
        self.register_free(&expr);
        let key = ParamKey(self.entries.len() as u32);
        self.entries.push(expr.clone());
        self.lookup.insert(expr, key);
        key
    }

    /// Fold an expression's symbols into `free`, keeping it deduplicated and sorted.
    ///
    /// Insertion is by binary search rather than push-then-sort because
    /// [`ParameterExpression::iter_symbols`] walks a `HashMap`: anything that took its order from that
    /// iteration would come out differently run to run. It is the same trap as building a
    /// [`Partition`](crate::partition::Partition) from a `HashSet`, which was a real bug in the
    /// original builder, and it is invisible in a single run.
    fn register_free(&mut self, expr: &ParameterExpression) {
        for symbol in expr.iter_symbols() {
            let key = ordering(symbol);
            if let Err(position) = self
                .free
                .binary_search_by(|existing| ordering(existing).cmp(&key))
            {
                self.free.insert(position, symbol.clone());
            }
        }
    }

    /// Look up an entry by key.
    pub fn get(&self, key: ParamKey) -> Option<&Arc<ParameterExpression>> {
        self.entries.get(key.index())
    }

    /// All entries, in key order.
    pub fn entries(&self) -> &[Arc<ParameterExpression>] {
        &self.entries
    }

    /// The symbols a caller must bind before sampling, deduplicated and in a stable order.
    pub fn free(&self) -> &[Symbol] {
        &self.free
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[pymethods]
impl ParameterTable {
    fn __len__(&self) -> usize {
        self.len()
    }

    fn __repr__(&self) -> String {
        let entries = self
            .entries
            .iter()
            .enumerate()
            .map(|(index, expr)| format!("{index}: {expr}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("ParameterTable({entries})")
    }

    /// The table's entries as expression strings, in key order.
    #[pyo3(name = "entries")]
    fn py_entries(&self) -> Vec<String> {
        self.entries.iter().map(|expr| expr.to_string()).collect()
    }

    /// The names of the parameters a caller must bind before sampling.
    #[getter]
    fn free_parameters(&self) -> Vec<String> {
        self.free
            .iter()
            .map(|symbol| symbol.fullname().into_owned())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh symbol. Two calls with the same name give two *different* parameters, exactly as two
    /// `Parameter("t")` objects do, so tests that mean one symbol have to reuse one of these.
    fn symbol(name: &str) -> Symbol {
        Symbol::standalone(name.to_string(), None)
    }

    fn expr(symbol: &Symbol) -> Arc<ParameterExpression> {
        Arc::new(ParameterExpression::from_symbol(symbol.clone()))
    }

    fn symbolic(name: &str) -> Arc<ParameterExpression> {
        expr(&symbol(name))
    }

    #[test]
    fn test_intern_dedups_equal_expressions() {
        let mut table = ParameterTable::new();
        let expr = symbolic("t");
        let a = table.intern(expr.clone());
        // A structurally equal expression, not the same `Arc`: equality is by expression, not identity.
        let b = table.intern(Arc::new((*expr).clone()));
        assert_eq!(a, b);
        assert_eq!(table.len(), 1);
        assert_eq!(table.free_parameters(), vec!["t".to_string()]);
    }

    #[test]
    fn test_intern_distinguishes_expressions() {
        let mut table = ParameterTable::new();
        let a = table.intern(symbolic("t"));
        let b = table.intern(symbolic("u"));
        assert_ne!(a, b);
        assert_eq!(table.len(), 2);
        assert_eq!(
            table.get(a).map(|expr| expr.to_string()),
            Some("t".to_string())
        );
    }

    #[test]
    fn test_free_is_sorted_regardless_of_intern_order() {
        // The guard on `iter_symbols` walking a `HashMap`: were `free` built in iteration order, this
        // would pass or fail depending on the run rather than on the code.
        let mut forwards = ParameterTable::new();
        forwards.intern(symbolic("alpha"));
        forwards.intern(symbolic("beta"));
        forwards.intern(symbolic("gamma"));

        let mut backwards = ParameterTable::new();
        backwards.intern(symbolic("gamma"));
        backwards.intern(symbolic("beta"));
        backwards.intern(symbolic("alpha"));

        assert_eq!(
            forwards.free_parameters(),
            vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()]
        );
        assert_eq!(forwards.free_parameters(), backwards.free_parameters());
    }

    #[test]
    fn test_a_multi_symbol_expression_contributes_every_symbol() {
        let mut table = ParameterTable::new();
        let sum = symbolic("y").add(&symbolic("x")).unwrap();
        table.intern(Arc::new(sum));
        assert_eq!(table.len(), 1);
        assert_eq!(
            table.free_parameters(),
            vec!["x".to_string(), "y".to_string()]
        );
    }

    #[test]
    fn test_free_dedups_across_entries() {
        // Two expressions over the same symbol are two entries but one thing to bind.
        let mut table = ParameterTable::new();
        let t = expr(&symbol("t"));
        table.intern(t.clone());
        table.intern(Arc::new(t.mul(&t).unwrap()));
        assert_eq!(table.len(), 2);
        assert_eq!(table.free_parameters(), vec!["t".to_string()]);
    }

    #[test]
    fn test_same_name_different_identity_both_appear() {
        // Two `Parameter("t")` objects are distinct parameters. Deduplicating them by name would
        // silently let one stand in for the other, so both are listed.
        let mut table = ParameterTable::new();
        table.intern(symbolic("t"));
        table.intern(symbolic("t"));
        assert_eq!(table.len(), 2);
        assert_eq!(table.free().len(), 2);
    }

    #[test]
    fn test_empty_table_has_nothing_to_bind() {
        let table = ParameterTable::new();
        assert!(table.is_empty());
        assert!(table.free_parameters().is_empty());
    }
}
