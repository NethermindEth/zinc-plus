use zip_plus::merkle_poc::Blake3MerkleTree;

// Enhanced main function for rigorous testing
// Note: This code assumes the Blake3MerkleTree and Blake3MerkleProof structs
// and their methods (new, prove, verify, root) are implemented
// according to the "Hash-then-Merkelize" strategy.
fn main() {
    // Test the original case (7 elements, asymmetric)
    let leaves = vec![
        "apple",
        "banana",
        "cherry",
        "date",
        "elderberry",
        "fig",
        "grape",
    ];
    test_merkle_tree(leaves);

    // Test edge cases and various sizes
    test_merkle_tree(vec![]); // 0
    test_merkle_tree(vec!["one"]); // 1
    test_merkle_tree(vec!["one", "two"]); // 2
    test_merkle_tree(vec!["one", "two", "three"]); // 3

    // Test with large and variable data sizes.
    // The "Hash-then-Merkelize" approach handles this efficiently.
    // Create 100 elements with varying lengths (from small to >1KB).
    let long_list: Vec<String> = (0..100)
        .map(|i| format!("data_{:04}_", i).repeat(i * 10 + 1))
        .collect();
    let long_list_refs: Vec<&str> = long_list.iter().map(|s| s.as_str()).collect();
    test_merkle_tree(long_list_refs);

    println!("\nAll tests passed successfully!");
}

fn test_merkle_tree(leaves: Vec<&str>) {
    println!(
        "\n--- Testing Merkle tree with {} elements (Hash-then-Merkelize) ---",
        leaves.len()
    );

    // 1. Build the tree.
    let merkle_tree = Blake3MerkleTree::new(&leaves);
    // Assuming a root() accessor method is implemented.
    let root_hash = &merkle_tree.root;

    println!("  -> Tree Root Hash: {}", root_hash);

    // NOTE: We specifically removed the assertion comparing this root to the
    // standard BLAKE3 hash of the serialized data. In this approach, they are
    // expected to be different.

    // 2. Generate and verify proof for every element.
    for i in 0..leaves.len() {
        // The actual data of the element we are proving.
        let element_data = leaves[i].as_bytes();

        let proof = merkle_tree.prove(i).expect("Failed to generate proof");

        // 3. Verify the proof (Success Case).
        // We must provide the element data during verification.
        let is_valid = proof.verify(root_hash, element_data);
        assert!(is_valid, "Verification failed for index {}", i);

        // 4. Verify failure with wrong root.
        let wrong_root_hash = blake3::hash(b"wrong root");
        assert!(
            !proof.verify(&wrong_root_hash, element_data),
            "Verification should fail for wrong root at index {}",
            i
        );

        // 5. Verify failure with tampered element data (Data Tampering).
        // We provide the correct proof and root, but incorrect data.
        // We create tampered data robustly to ensure it is always different.
        let mut tampered_data = element_data.to_vec();
        if !tampered_data.is_empty() {
            // Flip the first byte.
            tampered_data[0] = !tampered_data[0];
        } else {
            // If the element was empty, add a byte to make it non-empty.
            tampered_data.push(0xff);
        }

        assert!(
            !proof.verify(root_hash, &tampered_data),
            "Verification should fail for tampered data at index {}",
            i
        );

        // 6. Verify failure with tampered proof structure (Proof Tampering).
        if proof.total_elements > 1 {
            let mut tampered_proof = proof.clone();
            // Change the index to a different valid index.
            tampered_proof.element_index = (i + 1) % proof.total_elements;
            // Verification should fail because the path won't match the index geometry.
            assert!(
                !tampered_proof.verify(root_hash, element_data),
                "Verification should fail for wrong index at index {}",
                i
            );
        }
    }

    if !leaves.is_empty() {
        println!(
            "  -> All {} proofs generated and verified successfully.",
            leaves.len()
        );
    }
}
