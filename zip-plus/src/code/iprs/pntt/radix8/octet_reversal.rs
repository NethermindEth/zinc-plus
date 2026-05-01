/// Reverse octets (triples of consecutive binary digits)
/// in a given number `x`. The total number of octets
/// constituting the number is also provided as the argument
/// `num_octets`.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn octet_reversal(x: usize, num_octets: usize) -> usize {
    if num_octets == 0 {
        assert_eq!(x, 0, "With zero octets, the only valid input is 0, got {x}");
        return 0;
    }

    let mut result = 0;

    let mut left_octet_mask = 0b111;
    let mut right_octet_mask = 0b111 << (3 * (num_octets - 1));

    for i in 0..(num_octets / 2 + num_octets % 2) {
        let left_octet = x & left_octet_mask;
        let right_octet = x & right_octet_mask;

        result |= left_octet << (3 * (num_octets - 1 - 2 * i));
        result |= right_octet >> (3 * (num_octets - 1 - 2 * i));

        left_octet_mask <<= 3;
        right_octet_mask >>= 3;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::arithmetic_side_effects)]
    fn check_octet_reversal_self_inverse(num_octets: usize) {
        for x in 0..(1 << (3 * num_octets)) {
            assert_eq!(x, octet_reversal(octet_reversal(x, num_octets), num_octets));
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // long running
    fn octet_reversal_self_inverse() {
        for num_octets in 0..8 {
            check_octet_reversal_self_inverse(num_octets);
        }
    }

    #[test]
    fn octet_reversal_zero_octets() {
        assert_eq!(octet_reversal(0, 0), 0);
    }

    #[test]
    fn octet_reversal_one_octet_doesnt_do_anything() {
        for i in 0..(1 << 3) {
            assert_eq!(octet_reversal(i, 1), i);
        }
    }

    #[test]
    fn octet_reversal_known_result() {
        assert_eq!(
            octet_reversal(0b101_111_010_000_110, 5),
            0b110_000_010_111_101
        );
        assert_eq!(
            octet_reversal(0b101_111_010_000_110_000, 6),
            0b000_110_000_010_111_101
        );
    }
}
