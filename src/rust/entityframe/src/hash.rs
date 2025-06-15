use blake3::Hasher as Blake3Hasher;
use digest::{Digest, DynDigest};
use pyo3::prelude::*;
use sha2::{Sha256, Sha512};
use sha3::{Sha3_256, Sha3_512};

/// BLAKE3 wrapper to implement DynDigest trait for unified API
struct Blake3Wrapper {
    hasher: Blake3Hasher,
}

impl Blake3Wrapper {
    fn new() -> Self {
        Self {
            hasher: Blake3Hasher::new(),
        }
    }
}

impl digest::Update for Blake3Wrapper {
    fn update(&mut self, input: &[u8]) {
        self.hasher.update(input);
    }
}

impl digest::FixedOutput for Blake3Wrapper {
    fn finalize_into(self, out: &mut digest::Output<Self>) {
        let result = self.hasher.finalize();
        out.copy_from_slice(result.as_bytes());
    }
}

impl digest::OutputSizeUser for Blake3Wrapper {
    type OutputSize = digest::consts::U32; // BLAKE3 outputs 32 bytes by default
}

impl digest::HashMarker for Blake3Wrapper {}

impl DynDigest for Blake3Wrapper {
    fn update(&mut self, input: &[u8]) {
        digest::Update::update(self, input);
    }

    fn finalize_reset(&mut self) -> Box<[u8]> {
        let result = self.hasher.finalize();
        self.hasher.reset();
        result.as_bytes().to_vec().into_boxed_slice()
    }

    fn finalize(self: Box<Self>) -> Box<[u8]> {
        let result = self.hasher.finalize();
        result.as_bytes().to_vec().into_boxed_slice()
    }

    fn finalize_into(self, out: &mut [u8]) -> Result<(), digest::InvalidBufferSize> {
        let result = self.hasher.finalize();
        let output = result.as_bytes();
        if out.len() != output.len() {
            return Err(digest::InvalidBufferSize);
        }
        out.copy_from_slice(output);
        Ok(())
    }

    fn finalize_into_reset(&mut self, out: &mut [u8]) -> Result<(), digest::InvalidBufferSize> {
        let result = self.hasher.finalize();
        self.hasher.reset();
        let output = result.as_bytes();
        if out.len() != output.len() {
            return Err(digest::InvalidBufferSize);
        }
        out.copy_from_slice(output);
        Ok(())
    }

    fn reset(&mut self) {
        self.hasher.reset();
    }

    fn output_size(&self) -> usize {
        32 // BLAKE3 default output size
    }

    fn box_clone(&self) -> Box<dyn DynDigest> {
        Box::new(Blake3Wrapper::new())
    }
}

/// Create a hasher instance for the given algorithm
pub fn create_hasher(algorithm: &str) -> PyResult<Box<dyn DynDigest>> {
    match algorithm.to_lowercase().as_str() {
        "sha256" => Ok(Box::new(Sha256::new())),
        "sha512" => Ok(Box::new(Sha512::new())),
        "sha3-256" => Ok(Box::new(Sha3_256::new())),
        "sha3-512" => Ok(Box::new(Sha3_512::new())),
        "blake3" => Ok(Box::new(Blake3Wrapper::new())),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported hash algorithm: {}. Supported: sha256, sha512, sha3-256, sha3-512, blake3",
            algorithm
        ))),
    }
}

/// Collect all unique string IDs needed for a batch of entities
pub fn collect_string_ids(
    entities: &[&crate::entity::Entity],
    sorted_dataset_ids: &[u32],
) -> Vec<u32> {
    let mut string_ids = std::collections::HashSet::new();

    // Add all dataset IDs
    for &dataset_id in sorted_dataset_ids {
        string_ids.insert(dataset_id);
    }

    // Add all record IDs from all entities
    for entity in entities {
        for dataset_id in entity.get_dataset_ids() {
            string_ids.insert(dataset_id);

            if let Some(sorted_records) = entity.get_sorted_records(dataset_id) {
                for &record_id in sorted_records {
                    string_ids.insert(record_id);
                }
            } else {
                // Fallback: get record IDs from bitmap
                for record_id in entity.get_records_by_id(dataset_id) {
                    string_ids.insert(record_id);
                }
            }
        }
    }

    string_ids.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_hash_algorithms() {
        // Test that all algorithms produce different outputs for same input
        let test_data = b"hello world";

        let algorithms = ["sha256", "sha512", "sha3-256", "sha3-512", "blake3"];
        let mut hashes = Vec::new();

        for algorithm in &algorithms {
            let mut hasher = create_hasher(algorithm).unwrap();
            hasher.update(test_data);
            let hash = hasher.finalize().to_vec();
            hashes.push((algorithm, hash));
        }

        // Verify all hashes are different
        for i in 0..hashes.len() {
            for j in i + 1..hashes.len() {
                assert_ne!(
                    hashes[i].1, hashes[j].1,
                    "Algorithms {} and {} produced same hash",
                    hashes[i].0, hashes[j].0
                );
            }
        }

        // Verify expected output sizes
        assert_eq!(hashes[0].1.len(), 32); // SHA-256
        assert_eq!(hashes[1].1.len(), 64); // SHA-512
        assert_eq!(hashes[2].1.len(), 32); // SHA3-256
        assert_eq!(hashes[3].1.len(), 64); // SHA3-512
        assert_eq!(hashes[4].1.len(), 32); // BLAKE3
    }

    #[test]
    fn test_hash_determinism() {
        // Test that same input produces same hash
        let test_data = b"deterministic test";

        for algorithm in ["sha256", "blake3", "sha3-256"] {
            let hash1 = {
                let mut hasher = create_hasher(algorithm).unwrap();
                hasher.update(test_data);
                hasher.finalize().to_vec()
            };

            let hash2 = {
                let mut hasher = create_hasher(algorithm).unwrap();
                hasher.update(test_data);
                hasher.finalize().to_vec()
            };

            assert_eq!(hash1, hash2, "Algorithm {} not deterministic", algorithm);
        }
    }

    #[test]
    fn test_invalid_algorithm() {
        let result = create_hasher("invalid_algorithm");
        assert!(result.is_err());
    }
}
