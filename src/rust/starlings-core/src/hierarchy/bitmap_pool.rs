/// High-performance memory pool for RoaringBitmaps
///
/// Reduces allocation overhead for large-scale entity resolution workloads by reusing
/// RoaringBitmap instances. Optimized for 100M+ edge processing with minimal GC pressure.
use roaring::RoaringBitmap;
use std::collections::VecDeque;
use std::sync::Mutex;

/// Size class for bitmap pool classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PoolSizeClass {
    Small,
    Medium,
    Large,
}

/// Memory pool for efficient RoaringBitmap allocation and reuse
///
/// Features:
/// - Object pooling to reduce GC pressure
/// - Thread-safe with minimal contention
/// - Automatic cleanup to prevent memory leaks
/// - Size-based pooling for optimal cache locality
#[derive(Debug)]
pub struct BitmapPool {
    /// Pool of available bitmaps, grouped by expected size class
    small_pool: Mutex<VecDeque<RoaringBitmap>>, // For < 100 elements
    medium_pool: Mutex<VecDeque<RoaringBitmap>>, // For 100-10k elements
    large_pool: Mutex<VecDeque<RoaringBitmap>>,  // For > 10k elements

    /// Pool configuration
    max_small: usize,
    max_medium: usize,
    max_large: usize,
}

impl BitmapPool {
    /// Default pool sizes optimised for entity resolution workloads
    const DEFAULT_SMALL_POOL_SIZE: usize = 1000; // Many small entities
    const DEFAULT_MEDIUM_POOL_SIZE: usize = 100; // Fewer medium entities
    const DEFAULT_LARGE_POOL_SIZE: usize = 10; // Few large entities

    /// Size thresholds for pool classification
    const SMALL_THRESHOLD: u32 = 100;
    const MEDIUM_THRESHOLD: u32 = 10_000;

    /// Create a new bitmap pool with default configuration
    pub fn new() -> Self {
        Self::with_capacity(
            Self::DEFAULT_SMALL_POOL_SIZE,
            Self::DEFAULT_MEDIUM_POOL_SIZE,
            Self::DEFAULT_LARGE_POOL_SIZE,
        )
    }

    /// Create a bitmap pool with custom capacity limits
    pub fn with_capacity(max_small: usize, max_medium: usize, max_large: usize) -> Self {
        Self {
            small_pool: Mutex::new(VecDeque::with_capacity(max_small)),
            medium_pool: Mutex::new(VecDeque::with_capacity(max_medium)),
            large_pool: Mutex::new(VecDeque::with_capacity(max_large)),
            max_small,
            max_medium,
            max_large,
        }
    }

    /// Get a RoaringBitmap from the pool, creating new one if pool is empty
    ///
    /// The size_hint helps select the appropriate pool for better performance.
    ///
    /// # Arguments
    /// * `size_hint` - Expected number of elements (used for pool selection)
    ///
    /// # Returns
    /// A cleared RoaringBitmap ready for use along with its size class
    pub fn get(&self, size_hint: u32) -> (RoaringBitmap, PoolSizeClass) {
        let size_class = if size_hint < Self::SMALL_THRESHOLD {
            PoolSizeClass::Small
        } else if size_hint < Self::MEDIUM_THRESHOLD {
            PoolSizeClass::Medium
        } else {
            PoolSizeClass::Large
        };

        let pool = match size_class {
            PoolSizeClass::Small => &self.small_pool,
            PoolSizeClass::Medium => &self.medium_pool,
            PoolSizeClass::Large => &self.large_pool,
        };

        // Try to get from appropriate pool
        if let Ok(mut guard) = pool.lock() {
            if let Some(mut bitmap) = guard.pop_front() {
                bitmap.clear(); // Ensure clean state
                return (bitmap, size_class);
            }
        }

        // Pool empty or lock failed - create new bitmap
        (RoaringBitmap::new(), size_class)
    }

    /// Return a RoaringBitmap to the pool for reuse
    ///
    /// The bitmap will be cleared and returned to the pool based on its original size class.
    /// If the pool is full, the bitmap is dropped to prevent unbounded growth.
    ///
    /// # Arguments  
    /// * `mut bitmap` - The bitmap to return (will be cleared)
    /// * `size_class` - The original size class this bitmap was obtained for
    pub fn put(&self, mut bitmap: RoaringBitmap, size_class: PoolSizeClass) {
        bitmap.clear(); // Clear before storing

        let (pool, max_size) = match size_class {
            PoolSizeClass::Small => (&self.small_pool, self.max_small),
            PoolSizeClass::Medium => (&self.medium_pool, self.max_medium),
            PoolSizeClass::Large => (&self.large_pool, self.max_large),
        };

        // Try to return to appropriate pool
        if let Ok(mut guard) = pool.lock() {
            if guard.len() < max_size {
                guard.push_back(bitmap);
            }
            // If pool is full, bitmap is dropped automatically
        }
        // If lock failed, bitmap is dropped automatically
    }

    /// Get current pool statistics for monitoring
    pub fn stats(&self) -> BitmapPoolStats {
        let small_count = self.small_pool.lock().map_or(0, |guard| guard.len());
        let medium_count = self.medium_pool.lock().map_or(0, |guard| guard.len());
        let large_count = self.large_pool.lock().map_or(0, |guard| guard.len());

        BitmapPoolStats {
            small_available: small_count,
            medium_available: medium_count,
            large_available: large_count,
            total_available: small_count + medium_count + large_count,
        }
    }

    /// Clear all pools to free memory
    pub fn clear(&self) {
        if let Ok(mut guard) = self.small_pool.lock() {
            guard.clear();
        }
        if let Ok(mut guard) = self.medium_pool.lock() {
            guard.clear();
        }
        if let Ok(mut guard) = self.large_pool.lock() {
            guard.clear();
        }
    }
}

impl Default for BitmapPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for BitmapPool {
    fn clone(&self) -> Self {
        // Create new pool with same configuration
        Self::with_capacity(self.max_small, self.max_medium, self.max_large)
    }
}

/// Pool statistics for monitoring and debugging
#[derive(Debug, Clone)]
pub struct BitmapPoolStats {
    pub small_available: usize,
    pub medium_available: usize,
    pub large_available: usize,
    pub total_available: usize,
}

/// RAII wrapper for automatic bitmap return to pool
///
/// Ensures bitmaps are always returned to the pool when dropped,
/// preventing memory leaks in complex control flow.
pub struct PooledBitmap<'a> {
    bitmap: Option<RoaringBitmap>,
    pool: &'a BitmapPool,
    size_class: PoolSizeClass,
}

impl<'a> PooledBitmap<'a> {
    /// Create a new pooled bitmap
    pub fn new(pool: &'a BitmapPool, size_hint: u32) -> Self {
        let (bitmap, size_class) = pool.get(size_hint);
        Self {
            bitmap: Some(bitmap),
            pool,
            size_class,
        }
    }

    /// Get a reference to the underlying bitmap
    pub fn bitmap(&self) -> &RoaringBitmap {
        self.bitmap
            .as_ref()
            .expect("Bitmap should always be present")
    }

    /// Get a mutable reference to the underlying bitmap
    pub fn bitmap_mut(&mut self) -> &mut RoaringBitmap {
        self.bitmap
            .as_mut()
            .expect("Bitmap should always be present")
    }

    /// Take ownership of the bitmap, preventing automatic return to pool
    pub fn take(mut self) -> RoaringBitmap {
        self.bitmap.take().expect("Bitmap should always be present")
    }
}

impl Drop for PooledBitmap<'_> {
    fn drop(&mut self) {
        if let Some(bitmap) = self.bitmap.take() {
            self.pool.put(bitmap, self.size_class);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitmap_pool_basic() {
        let pool = BitmapPool::new();

        // Get a bitmap from empty pool
        let (mut bitmap1, size_class1) = pool.get(50);
        assert_eq!(bitmap1.len(), 0);
        assert_eq!(size_class1, PoolSizeClass::Small);

        // Add some data
        bitmap1.insert(1);
        bitmap1.insert(2);
        bitmap1.insert(3);
        assert_eq!(bitmap1.len(), 3);

        // Return to pool
        pool.put(bitmap1, size_class1);

        // Get another bitmap - should be reused and cleared
        let (bitmap2, _) = pool.get(50);
        assert_eq!(bitmap2.len(), 0);
    }

    #[test]
    fn test_pool_size_classification() {
        let pool = BitmapPool::new();

        // Test small pool
        let (small_bitmap, small_class) = pool.get(10);
        assert_eq!(small_class, PoolSizeClass::Small);
        pool.put(small_bitmap, small_class);

        // Test medium pool
        let (medium_bitmap, medium_class) = pool.get(1000);
        assert_eq!(medium_class, PoolSizeClass::Medium);
        pool.put(medium_bitmap, medium_class);

        // Test large pool
        let (large_bitmap, large_class) = pool.get(50000);
        assert_eq!(large_class, PoolSizeClass::Large);
        pool.put(large_bitmap, large_class);

        let stats = pool.stats();
        assert_eq!(stats.small_available, 1);
        assert_eq!(stats.medium_available, 1);
        assert_eq!(stats.large_available, 1);
        assert_eq!(stats.total_available, 3);
    }

    #[test]
    fn test_pool_capacity_limits() {
        let pool = BitmapPool::with_capacity(2, 1, 1);

        // Create 3 separate bitmaps and populate them to make them distinct
        let (mut bitmap1, class1) = pool.get(10);
        bitmap1.insert(1);
        let (mut bitmap2, class2) = pool.get(10);
        bitmap2.insert(2);
        let (mut bitmap3, class3) = pool.get(10);
        bitmap3.insert(3);

        // Return them all to pool - only 2 should be kept
        pool.put(bitmap1, class1);
        pool.put(bitmap2, class2);
        pool.put(bitmap3, class3); // This should be dropped

        let stats = pool.stats();
        assert_eq!(stats.small_available, 2); // Only 2, third was dropped
    }

    #[test]
    fn test_pooled_bitmap_raii() {
        let pool = BitmapPool::new();

        {
            let mut pooled = PooledBitmap::new(&pool, 100);
            pooled.bitmap_mut().insert(42);
            assert!(pooled.bitmap().contains(42));
            // Drops here, should return to pool
        }

        // Get another bitmap - should be reused
        let (bitmap, _) = pool.get(100);
        assert_eq!(bitmap.len(), 0); // Should be cleared

        let stats = pool.stats();
        assert_eq!(stats.total_available, 0); // One bitmap is checked out
    }

    #[test]
    fn test_pooled_bitmap_take() {
        let pool = BitmapPool::new();

        let pooled = PooledBitmap::new(&pool, 100);
        let bitmap = pooled.take(); // Prevents return to pool

        assert_eq!(bitmap.len(), 0);

        let stats = pool.stats();
        assert_eq!(stats.total_available, 0); // Bitmap was taken, not returned
    }

    #[test]
    fn test_pool_clear() {
        let pool = BitmapPool::new();

        // Add bitmaps to all pools
        let (bitmap1, class1) = pool.get(10);
        pool.put(bitmap1, class1); // small
        let (bitmap2, class2) = pool.get(1000);
        pool.put(bitmap2, class2); // medium
        let (bitmap3, class3) = pool.get(50000);
        pool.put(bitmap3, class3); // large

        let stats_before = pool.stats();
        assert_eq!(stats_before.total_available, 3);

        pool.clear();

        let stats_after = pool.stats();
        assert_eq!(stats_after.total_available, 0);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let pool = Arc::new(BitmapPool::new());
        let mut handles = vec![];

        // Spawn multiple threads using the pool
        for i in 0..10 {
            let pool_clone = Arc::clone(&pool);
            let handle = thread::spawn(move || {
                let (mut bitmap, size_class) = pool_clone.get(i * 100);
                bitmap.insert(i);
                assert!(bitmap.contains(i));
                pool_clone.put(bitmap, size_class);
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Pool should have bitmaps available
        let stats = pool.stats();
        assert!(stats.total_available > 0);
    }
}
