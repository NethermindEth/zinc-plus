#[allow(dead_code)]
fn hello_world() -> u32 {
    2 + 2
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(super::hello_world(), 4);
    }
}
