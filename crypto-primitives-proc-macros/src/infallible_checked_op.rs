use syn::{
    Attribute, Expr, ExprLit, ExprPath, ExprTuple, Ident, Lit, Path, Result as SynResult,
    punctuated::Punctuated, spanned::Spanned, token::Comma,
};

#[derive(Debug)]
pub(super) enum Op {
    Unary {
        trait_path: Path,
        method: Ident,
    },
    Binary {
        trait_path: Path,
        method: Ident,
        rhs_by_value: bool,
    },
}

/// Parse attribute arguments that are a comma-separated list of 2- or 3-element
/// tuples:   unary:  (TraitPath, method), (OtherTrait, other_method)
///   binary: (TraitPath, method), (OtherTrait, other_method, true)
///
/// Both trait and method are *identifiers/paths* (no string literals).
pub(super) fn parse_pairs_from_attr(attr: &Attribute, expect_binary: bool) -> SynResult<Vec<Op>> {
    // parse a comma-separated list of expressions (each expected to be a tuple
    // expression)
    let exprs: Punctuated<Expr, Comma> =
        attr.parse_args_with(Punctuated::<Expr, Comma>::parse_terminated)?;

    let mut out = Vec::new();

    for expr in exprs.into_iter() {
        // each expr must be a tuple expression: (a, b) or (a, b, c)
        let tuple = match expr {
            Expr::Tuple(ExprTuple { elems, .. }) => elems,
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "expected tuple like `(TraitPath, method)`",
                ));
            }
        };

        // convert the elems into a Vec<Expr>
        let elems_vec: Vec<Expr> = tuple.into_iter().collect();

        if elems_vec.len() < 2 || elems_vec.len() > 3 {
            return Err(syn::Error::new(
                attr.span(),
                "expected tuple of 2 (unary) or 2-3 (binary) elements: (TraitPath, method[, by_value])",
            ));
        }

        // first element: trait path (Expr::Path)
        let trait_path = match &elems_vec[0] {
            Expr::Path(ExprPath { path, .. }) => path.clone(),
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "expected trait path (e.g. crate::traits::CheckedAdd)",
                ));
            }
        };

        // second element: method identifier as a path whose last segment is the ident
        let method_ident = match &elems_vec[1] {
            Expr::Path(ExprPath { path, .. }) => {
                if let Some(seg) = path.segments.last() {
                    seg.ident.clone()
                } else {
                    return Err(syn::Error::new_spanned(
                        path,
                        "expected method identifier (e.g. add)",
                    ));
                }
            }
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "expected method identifier (e.g. add)",
                ));
            }
        };

        if expect_binary {
            // optional third element: bool literal (rhs_by_value)
            let rhs_by_value = if elems_vec.len() == 3 {
                match &elems_vec[2] {
                    Expr::Lit(ExprLit {
                        lit: Lit::Bool(lb), ..
                    }) => lb.value,
                    other => {
                        return Err(syn::Error::new_spanned(
                            other,
                            "expected boolean literal as third tuple element (true/false)",
                        ));
                    }
                }
            } else {
                false
            };

            out.push(Op::Binary {
                trait_path,
                method: method_ident,
                rhs_by_value,
            });
        } else {
            // unary; there must be exactly 2 elements
            if elems_vec.len() != 2 {
                return Err(syn::Error::new(
                    attr.span(),
                    "unary attribute tuples must have exactly 2 elements: (TraitPath, method)",
                ));
            }
            out.push(Op::Unary {
                trait_path,
                method: method_ident,
            });
        }
    }

    Ok(out)
}
