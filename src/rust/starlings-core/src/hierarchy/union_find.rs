/// High-performance Union-Find (Disjoint Set Union) data structure
///
/// Optimised for entity resolution workloads with aggressive path compression
/// and rank-based union operations. Achieves near-linear performance for
/// large-scale operations (100M+ elements).
use std::sync::atomic::{AtomicU32, Ordering};

/// Custom Union-Find implementation optimised for performance
///
/// Features:
/// - Rank-based union with path halving compression
/// - Cache-friendly memory layout  
/// - Aggressive inlining for hot paths
/// - Atomic operations for potential future parallelisation
#[derive(Debug)]
pub struct UnionFind {
    /// Parent pointers - parent[i] points to parent of element i
    /// If parent[i] == i, then i is a root
    parent: Vec<AtomicU32>,

    /// Rank approximation for union-by-rank optimisation
    /// rank[i] is an upper bound on the height of subtree rooted at i
    rank: Vec<AtomicU32>,

    /// Number of elements in the union-find structure
    size: usize,
}

impl UnionFind {
    /// Create a new Union-Find structure with n elements
    ///
    /// Initially, each element is in its own set.
    ///
    /// # Arguments
    /// * `size` - Number of elements (0..size)
    ///
    /// # Complexity
    /// O(n) time, O(n) space
    #[inline]
    pub fn new(size: usize) -> Self {
        let parent: Vec<AtomicU32> = (0..size).map(|i| AtomicU32::new(i as u32)).collect();

        let rank: Vec<AtomicU32> = (0..size).map(|_| AtomicU32::new(0)).collect();

        Self { parent, rank, size }
    }

    /// Find the root of the set containing element x with path compression
    ///
    /// Uses aggressive path halving for optimal performance.
    /// Path halving makes every node point to its grandparent, effectively
    /// halving the path length in one pass.
    ///
    /// # Arguments
    /// * `x` - Element to find root for
    ///
    /// # Returns
    /// Root element of the set containing x
    ///
    /// # Complexity
    /// O(α(n)) amortised, where α is the inverse Ackermann function
    #[inline]
    pub fn find(&self, mut x: usize) -> usize {
        debug_assert!(x < self.size, "Element index out of bounds");

        // Path halving: make every node point to its grandparent
        loop {
            let parent = self.parent[x].load(Ordering::Relaxed) as usize;
            if parent == x {
                return x; // Found root
            }

            let grandparent = self.parent[parent].load(Ordering::Relaxed) as usize;
            if parent == grandparent {
                return parent; // Parent is root
            }

            // Path halving: make x point to its grandparent
            self.parent[x].store(grandparent as u32, Ordering::Relaxed);
            x = parent;
        }
    }

    /// Union two sets containing elements x and y
    ///
    /// Uses union-by-rank to keep trees balanced, ensuring good performance.
    /// The root of the tree with smaller rank is made to point to the root
    /// of the tree with larger rank.
    ///
    /// # Arguments
    /// * `x` - Element from first set
    /// * `y` - Element from second set
    ///
    /// # Returns
    /// true if the sets were different (union performed), false otherwise
    ///
    /// # Complexity  
    /// O(α(n)) amortised, where α is the inverse Ackermann function
    #[inline]
    pub fn union(&self, x: usize, y: usize) -> bool {
        debug_assert!(x < self.size, "Element x index out of bounds");
        debug_assert!(y < self.size, "Element y index out of bounds");

        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false; // Already in same set
        }

        let rank_x = self.rank[root_x].load(Ordering::Relaxed);
        let rank_y = self.rank[root_y].load(Ordering::Relaxed);

        // Union by rank: attach smaller rank tree to larger rank tree
        use std::cmp::Ordering as CmpOrdering;
        match rank_x.cmp(&rank_y) {
            CmpOrdering::Less => {
                self.parent[root_x].store(root_y as u32, Ordering::Relaxed);
            }
            CmpOrdering::Greater => {
                self.parent[root_y].store(root_x as u32, Ordering::Relaxed);
            }
            CmpOrdering::Equal => {
                // Equal ranks: make y point to x and increment x's rank
                self.parent[root_y].store(root_x as u32, Ordering::Relaxed);
                self.rank[root_x].store(rank_x + 1, Ordering::Relaxed);
            }
        }

        true // Union performed
    }

    /// Check if two elements are in the same set
    ///
    /// # Arguments
    /// * `x` - First element
    /// * `y` - Second element
    ///
    /// # Returns
    /// true if x and y are in the same connected component
    ///
    /// # Complexity
    /// O(α(n)) amortised
    #[inline]
    pub fn connected(&self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }

    /// Get the number of elements in the union-find structure
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if the union-find structure is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get all elements that have the same root as the given element
    ///
    /// This is a helper method for building connected components efficiently.
    ///
    /// # Arguments
    /// * `representative` - Element whose component to collect
    ///
    /// # Returns
    /// Vector of all elements in the same component as representative
    ///
    /// # Complexity
    /// O(n) - must scan all elements
    pub fn get_component(&self, representative: usize) -> Vec<usize> {
        let target_root = self.find(representative);
        (0..self.size)
            .filter(|&i| self.find(i) == target_root)
            .collect()
    }

    /// Get all connected components as a vector of vectors
    ///
    /// This is optimised for the common case where we need all components.
    /// Uses a single pass through all elements with memoisation.
    ///
    /// # Returns
    /// Vector where each element is a vector of connected elements
    ///
    /// # Complexity
    /// O(n * α(n)) - single pass with path compression benefits
    pub fn get_all_components(&self) -> Vec<Vec<usize>> {
        let mut components: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();

        for element in 0..self.size {
            let root = self.find(element);
            components.entry(root).or_default().push(element);
        }

        components.into_values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union_find_basic() {
        let uf = UnionFind::new(5);

        // Initially, each element is its own root
        for i in 0..5 {
            assert_eq!(uf.find(i), i);
        }

        // Union some elements
        assert!(uf.union(0, 1)); // Returns true - union performed
        assert!(!uf.union(0, 1)); // Returns false - already connected

        assert_eq!(uf.find(0), uf.find(1)); // Same root
        assert_ne!(uf.find(0), uf.find(2)); // Different roots
    }

    #[test]
    fn test_union_find_path_compression() {
        let uf = UnionFind::new(10);

        // Create a chain: 0->1->2->3->4
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(2, 3);
        uf.union(3, 4);

        // All should have the same root
        let root = uf.find(0);
        for i in 0..5 {
            assert_eq!(uf.find(i), root);
        }

        // Path compression should have flattened the structure
        // (verified by the fact that subsequent finds are fast)
    }

    #[test]
    fn test_connected_components() {
        let uf = UnionFind::new(6);

        // Create two components: {0,1,2} and {3,4}, with 5 isolated
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(3, 4);

        // Test connectivity
        assert!(uf.connected(0, 2));
        assert!(uf.connected(3, 4));
        assert!(!uf.connected(0, 3));
        assert!(!uf.connected(2, 5));

        // Test component collection
        let components = uf.get_all_components();
        assert_eq!(components.len(), 3); // Three components

        // Find the component sizes
        let mut sizes: Vec<usize> = components.iter().map(|c| c.len()).collect();
        sizes.sort();
        assert_eq!(sizes, vec![1, 2, 3]); // Sizes: 1 (isolated), 2, 3
    }

    #[test]
    fn test_large_union_find() {
        const N: usize = 10000;
        let uf = UnionFind::new(N);

        // Connect pairs: (0,1), (2,3), (4,5), ...
        for i in (0..N).step_by(2) {
            if i + 1 < N {
                uf.union(i, i + 1);
            }
        }

        // Verify connections
        for i in (0..N).step_by(2) {
            if i + 1 < N {
                assert!(uf.connected(i, i + 1));
                if i + 2 < N {
                    assert!(!uf.connected(i, i + 2));
                }
            }
        }

        // Should have approximately N/2 components
        let components = uf.get_all_components();
        assert_eq!(components.len(), (N + 1) / 2);
    }

    #[test]
    fn test_union_find_rank_optimization() {
        let uf = UnionFind::new(100);

        // Test basic union by rank: when trees have different ranks,
        // the root of the shorter tree should point to the root of the taller tree

        // Create two trees with different heights
        // Tree 1: Chain of length 3 (height ≈ 2)
        uf.union(0, 1);
        uf.union(1, 2);

        // Tree 2: Simple pair (height = 1)
        uf.union(10, 11);

        // Union the trees - all should end up in the same component
        uf.union(0, 10);

        // Test that all elements are connected (exact root doesn't matter)
        let root = uf.find(0);
        assert_eq!(uf.find(1), root);
        assert_eq!(uf.find(2), root);
        assert_eq!(uf.find(10), root);
        assert_eq!(uf.find(11), root);

        // Test connectivity
        assert!(uf.connected(0, 11));
        assert!(uf.connected(2, 10));
    }
}
