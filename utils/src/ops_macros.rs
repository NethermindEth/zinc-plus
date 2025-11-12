#[macro_export]
macro_rules! neg {
    ($a:expr) => {
        neg!($a, "Negation overflow")
    };
    ($a:expr, $msg:expr) => {
        $a.checked_neg().expect($msg)
    };
}

#[macro_export]
macro_rules! add {
    ($a:expr, $b:expr) => {
        add!($a, $b, "Addition overflow")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        $a.checked_add($b).expect($msg)
    };
}

#[macro_export]
macro_rules! sub {
    ($a:expr, $b:expr) => {
        sub!($a, $b, "Subtraction overflow")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        $a.checked_sub($b).expect($msg)
    };
}

#[macro_export]
macro_rules! mul {
    ($a:expr, $b:expr) => {
        mul!($a, $b, "Multiplication overflow")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        $a.checked_mul($b).expect($msg)
    };
}

#[macro_export]
macro_rules! div {
    ($a:expr, $b:expr) => {
        div!($a, $b, "Division by zero")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        $a.checked_div($b).expect($msg)
    };
}

#[macro_export]
macro_rules! rem {
    ($a:expr, $b:expr) => {
        rem!($a, $b, "Division by zero")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        $a.checked_rem($b).expect($msg)
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
