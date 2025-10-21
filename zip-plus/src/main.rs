use zip_plus::merkle_poc::Blake3MerkleTree;

// Enhanced main function for rigorous testing
fn main() {
    // Test the original case (7 elements, asymmetric)
    let leaves = vec!["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"];
    test_merkle_tree(leaves);

    // Test edge cases and various sizes
    test_merkle_tree(vec![]); // 0
    test_merkle_tree(vec!["one"]); // 1
    test_merkle_tree(vec!["one", "two"]); // 2
    test_merkle_tree(vec!["one", "two", "three"]); // 3

    // Test with data causing multiple chunks
    let long_list: Vec<String> = (0..100).map(|i| format!("data_{:04}_", i).repeat(20)).collect();
    let long_list_refs: Vec<&str> = long_list.iter().map(|s| s.as_str()).collect();
    test_merkle_tree(long_list_refs);

    println!("\nAll tests passed successfully!");
}

fn test_merkle_tree(leaves: Vec<&str>) {
    println!("\n--- Testing Merkle tree with {} elements ---", leaves.len());

    // 1. Build the tree.
    let merkle_tree = Blake3MerkleTree::new(&leaves);
    let root_hash = merkle_tree.root;

    // CRITICAL: Verify the root hash matches standard Blake3 hashing of the serialized data.
    // This confirms our manual construction exactly mimics Blake3's internal structure.
    let expected_root = blake3::hash(&merkle_tree.serialized_data);

    println!("  -> Tree Root Hash (Manual): {}", root_hash);
    println!("  -> Expected Hash (Standard): {}", expected_root);
    assert_eq!(root_hash, expected_root, "Manual construction root hash mismatch!");

    // 2. Generate and verify proof for every element.
    for i in 0..leaves.len() {
        let proof = merkle_tree.prove(i).expect("Failed to generate proof");

        // 3. Verify the proof.
        let is_valid = proof.verify(root_hash);
        assert!(is_valid, "Verification failed for index {}", i);

        // 4. Verify failure with wrong root.
        let wrong_root_hash = blake3::hash(b"wrong root");
        assert!(!proof.verify(&wrong_root_hash), "Verification should fail for wrong root");

        // 5. Verify failure with tampered data.
        let mut tampered_proof = proof.clone();
        // Flip the first byte of the chunk data.
        if !tampered_proof.chunk_bytes.is_empty() {
            tampered_proof.chunk_bytes[0] = !tampered_proof.chunk_bytes[0];
            assert!(!tampered_proof.verify(root_hash), "Verification should fail for tampered data");
        }
    }
    if !leaves.is_empty() {
        println!("  -> All {} proofs generated and verified successfully.", leaves.len());
    }
}
