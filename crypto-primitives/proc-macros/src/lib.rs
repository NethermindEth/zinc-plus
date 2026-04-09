//! This internal helper crate provides a procedural macro to derive
//! implementations of infallible checked operations for types that implement
//! the corresponding non-checked operations.
mod infallible_checked_op;

use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

/// In case checked operations (like `CheckedAdd` or `CheckedNeg`) can't fail
/// for a type, derive this macro to automatically implement them for type
/// calling non-checked operations under the hood.
///
/// Usage example:
/// ```rust
/// # use crypto_primitives_proc_macros::InfallibleCheckedOp;
/// # use num_traits::{CheckedAdd, CheckedSub, CheckedMul, CheckedNeg};
/// # use std::ops::{Add, Sub, Mul, Neg};
///
/// #[derive(Clone, InfallibleCheckedOp)]
/// #[infallible_checked_unary_op((CheckedNeg, neg))]
/// #[infallible_checked_binary_op(
///     (CheckedAdd, add, true), // Optional third argument - whether to pass `rhs` by value.
///     (CheckedSub, sub), // By default, `rhs` is passed by reference.
/// )]
/// pub struct MyField {};
///
/// impl Neg for MyField {
///     type Output = MyField;
///
///     fn neg(self) -> MyField {
///         todo!()
///     }
/// }
///
/// impl Add for MyField {
///     type Output = MyField;
///
///     fn add(self, rhs: MyField) -> MyField {
///         todo!()
///     }
/// }
///
/// # impl Sub for MyField {
/// #     type Output = MyField;
/// #     fn sub(self, rhs: MyField) -> MyField { todo!() }
/// # }
/// #
/// impl Sub<&Self> for MyField {
///     type Output = MyField;
///
///     fn sub(self, rhs: &MyField) -> MyField {
///         todo!()
///     }
/// }
/// ```
#[proc_macro_derive(
    InfallibleCheckedOp,
    attributes(infallible_checked_unary_op, infallible_checked_binary_op)
)]
pub fn derive_infallible_checked_op(input: TokenStream) -> TokenStream {
    use infallible_checked_op::*;
    let input = parse_macro_input!(input as DeriveInput);

    // collect all relevant attributes
    let mut ops: Vec<Op> = Vec::new();
    for attr in &input.attrs {
        if attr.path().is_ident("infallible_checked_unary_op") {
            match parse_pairs_from_attr(attr, false) {
                Ok(mut v) => ops.append(&mut v),
                Err(e) => return e.to_compile_error().into(),
            }
        } else if attr.path().is_ident("infallible_checked_binary_op") {
            match parse_pairs_from_attr(attr, true) {
                Ok(mut v) => ops.append(&mut v),
                Err(e) => return e.to_compile_error().into(),
            }
        }
    }

    if ops.is_empty() {
        return syn::Error::new_spanned(
            &input.ident,
            "no infallible_checked_* attributes found; attach at least one `#[infallible_checked_unary_op(...)]` or `#[infallible_checked_binary_op(...)]`",
        )
        .to_compile_error()
        .into();
    }

    let name = &input.ident;
    let generics = input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let mut impls = proc_macro2::TokenStream::new();

    for op in ops {
        match op {
            Op::Unary { trait_path, method } => {
                let checked_ident = Ident::new(&format!("checked_{}", method), method.span());
                let method_ident = method;

                let ts = quote! {
                    impl #impl_generics #trait_path for #name #ty_generics #where_clause {
                        fn #checked_ident(&self) -> Option<Self> {
                            Some(self.clone().#method_ident())
                        }
                    }
                };
                impls.extend(ts);
            }
            Op::Binary {
                trait_path,
                method,
                rhs_by_value,
            } => {
                let checked_ident = Ident::new(&format!("checked_{}", method), method.span());
                let method_ident = method;

                if rhs_by_value {
                    let ts = quote! {
                        impl #impl_generics #trait_path for #name #ty_generics #where_clause {
                            fn #checked_ident(&self, rhs: &Self) -> Option<Self> {
                                Some(self.clone().#method_ident(rhs.clone()))
                            }
                        }
                    };
                    impls.extend(ts);
                } else {
                    let ts = quote! {
                        impl #impl_generics #trait_path for #name #ty_generics #where_clause {
                            fn #checked_ident(&self, rhs: &Self) -> Option<Self> {
                                Some(self.clone().#method_ident(rhs))
                            }
                        }
                    };
                    impls.extend(ts);
                }
            }
        }
    }

    quote!(#impls).into()
}
