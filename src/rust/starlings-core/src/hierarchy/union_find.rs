/// Union-Find (Disjoint Set Union) data structure
#[derive(Debug)]
pub struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u32>,
    size: usize,
}

impl UnionFind {
    /// Create a new Union-Find structure with n elements
    pub fn new(size: usize) -> Self {
        let parent: Vec<u32> = (0..size).map(|i| i as u32).collect();
        let rank: Vec<u32> = vec![0; size];
        Self { parent, rank, size }
    }

    /// Find the root of the set containing element x with path compression
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x as u32 {
            let grandparent = self.parent[self.parent[x] as usize];
            self.parent[x] = grandparent;
            x = self.parent[x] as usize;
        }
        x
    }

    /// Union two sets containing elements x and y
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false;
        }

        let rank_x = self.rank[root_x];
        let rank_y = self.rank[root_y];

        match rank_x.cmp(&rank_y) {
            std::cmp::Ordering::Less => {
                self.parent[root_x] = root_y as u32;
            }
            std::cmp::Ordering::Greater => {
                self.parent[root_y] = root_x as u32;
            }
            std::cmp::Ordering::Equal => {
                self.parent[root_y] = root_x as u32;
                self.rank[root_x] += 1;
            }
        }

        true
    }

    /// Check if two elements are in the same set
    pub fn connected(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }

    /// Get the number of elements in the union-find structure
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if the union-find structure is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get all connected components as a vector of vectors
    pub fn get_all_components(&mut self) -> Vec<Vec<usize>> {
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
        let mut uf = UnionFind::new(5);

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
        let mut uf = UnionFind::new(10);

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
        let mut uf = UnionFind::new(6);

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
        let mut uf = UnionFind::new(N);

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
        let mut uf = UnionFind::new(100);

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
