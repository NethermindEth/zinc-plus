#[macro_export]
macro_rules! neg {
    ($a:expr) => {
        neg!($a, "Negation overflow")
    };
    ($a:expr, $msg:expr) => {
        num_traits::CheckedNeg::checked_neg(&$a).expect($msg)
    };
}

#[macro_export]
macro_rules! add {
    ($a:expr, $b:expr) => {
        add!($a, $b, "Addition overflow")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        num_traits::CheckedAdd::checked_add(&$a, &$b).expect($msg)
    };
}

#[macro_export]
macro_rules! sub {
    ($a:expr, $b:expr) => {
        sub!($a, $b, "Subtraction overflow")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        num_traits::CheckedSub::checked_sub(&$a, &$b).expect($msg)
    };
}

#[macro_export]
macro_rules! mul {
    ($a:expr, $b:expr) => {
        mul!($a, $b, "Multiplication overflow")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        num_traits::CheckedMul::checked_mul(&$a, &$b).expect($msg)
    };
}

#[macro_export]
macro_rules! div {
    ($a:expr, $b:expr) => {
        div!($a, $b, "Division by zero")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        num_traits::CheckedDiv::checked_div(&$a, &$b).expect($msg)
    };
}

#[macro_export]
macro_rules! rem {
    ($a:expr, $b:expr) => {
        rem!($a, $b, "Division by zero")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        num_traits::CheckedRem::checked_rem(&$a, &$b).expect($msg)
    };
}

#[macro_export]
macro_rules! ilog_round_up {
    ($a:expr, $tp: ty) => {{
        let res = if $a.is_power_of_two() {
            $a.ilog2()
        } else {
            add!($a.ilog2(), 1)
        };
        <$tp>::try_from(res).expect(concat!("ilog doesn't fit ", stringify!($tp)))
    }};
}
