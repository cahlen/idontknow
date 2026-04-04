# Exoo's R(5,5) ≥ 43 Circulant Coloring

## Citation
- **Geoffrey Exoo** (1989). "A lower bound for r(5, 5)"
- Verified and analyzed: **Meng et al.** (2022). "Study of Exoo's Lower Bound for Ramsey Number R(5,5)" – [arXiv:2212.12630](https://arxiv.org/abs/2212.12630)

## Graph Structure

| Property | Value |
|----------|-------|
| **Name** | Cyclic(43) |
| **Vertices** | 43 (labeled 0 to 42) |
| **Type** | Circulant graph on K₄₃ with RED/BLUE 2-coloring |
| **Coloring** | Edge colors depend only on vertex difference mod 43 |

## Difference Sets

### BLUE Edges
An edge (i, j) is **BLUE** if `(j - i) mod 43 ∈ S_blue`:

```
S_blue = {3, 4, 5, 6, 8, 9, 11, 15, 17, 19, 24, 26, 28, 32, 34, 35, 37, 38, 39, 40}
|S_blue| = 20
Degree in BLUE subgraph: 20 neighbors per vertex
```

### RED Edges
An edge (i, j) is **RED** if `(j - i) mod 43 ∈ S_red`:

```
S_red = {1, 2, 7, 10, 12, 13, 14, 16, 18, 20, 21, 22, 23, 25, 27, 29, 30, 31, 33, 36, 41, 42}
|S_red| = 22
Degree in RED subgraph: 22 neighbors per vertex
```

## Critical Property: Every K₅ is Red

**For every i ∈ {0, 1, ..., 42}, the 5-tuple**
```
(i, i+1, i+2, i+22, i+23) mod 43
```
**forms a complete RED K₅.**

### Examples
- **i=0**: vertices {0, 1, 2, 22, 23} – all edges RED
- **i=1**: vertices {1, 2, 3, 23, 24} – all edges RED
- **i=20**: vertices {20, 21, 22, 42, 0} – all edges RED
- ... (all 43 such tuples)

### Verification (i=0)
| Edge | Difference | Color |
|------|-----------|-------|
| 0–1 | 1 | RED ✓ |
| 0–2 | 2 | RED ✓ |
| 0–22 | 22 | RED ✓ |
| 0–23 | 23 | RED ✓ |
| 1–2 | 1 | RED ✓ |
| 1–22 | 21 | RED ✓ |
| 1–23 | 22 | RED ✓ |
| 2–22 | 20 | RED ✓ |
| 2–23 | 21 | RED ✓ |
| 22–23 | 1 | RED ✓ |

## Ramsey Theory Implication

This 2-coloring of K₄₃ satisfies:
- **BLUE subgraph is K₅-free** (no blue K₅ exists anywhere)
- **RED subgraph is K₅-free** (no red K₅ exists anywhere)

Therefore:
```
R(5,5) > 42  ⟹  R(5,5) ≥ 43
```

## Current Bounds

| Bound | Value | Source |
|-------|-------|--------|
| Lower | 43 | Exoo (1989) |
| Upper | 48 | McKay & Angeltveit (2024) |
| Conjecture | 43 | Community consensus |

## Implementation

### Python
```python
def is_red_edge(i, j, n=43):
    """Check if edge (i,j) is RED in Cyclic(43)"""
    diff = (j - i) % n
    S_red = {1, 2, 7, 10, 12, 13, 14, 16, 18, 20, 21, 22, 23,
             25, 27, 29, 30, 31, 33, 36, 41, 42}
    return diff in S_red

def is_blue_edge(i, j, n=43):
    """Check if edge (i,j) is BLUE in Cyclic(43)"""
    diff = (j - i) % n
    S_blue = {3, 4, 5, 6, 8, 9, 11, 15, 17, 19, 24, 26, 28,
              32, 34, 35, 37, 38, 39, 40}
    return diff in S_blue

def find_red_k5_for_i(i, n=43):
    """Return the guaranteed red K₅ for position i"""
    return [(i + x) % n for x in [0, 1, 2, 22, 23]]
```

### CUDA Kernel Pattern
```cuda
__device__ bool is_red_edge(int i, int j) {
    int diff = ((j - i) % 43 + 43) % 43;
    // Precomputed bitmask or lookup table for S_red
    __shared__ uint32_t S_red_mask[2];  // Can encode in 44 bits
    return (S_red_mask[diff / 32] >> (diff % 32)) & 1;
}
```

## Related Resources

1. **Explicit Adjacency Matrices** (GitHub):
   - [Two K₄₃ graphs with only two monochromatic K₅'s](https://gist.github.com/etherwalker/8d64fa0a1cc1dd508f75bf651aaec873)
   - Contains full adjacency matrix representations

2. **Exoo's Construction Archive**:
   - [Ramsey Constructions @ Indiana State](https://cs.indstate.edu/ge/RAMSEY/)
   - Includes various Ramsey graph formats

3. **McKay's Ramsey Data**:
   - [Brendan McKay's Ramsey Numbers Data](https://users.cecs.anu.edu.au/~bdm/data/ramsey.html)
   - Links to 656 non-isomorphic (5,5,42)-graphs

## Variant: EXOO(42)

Exoo(42) uses the 42 vertices {0,1,...,41} with modified color rules for differences {1, 21, 22, 42} relative to Cyclic(43). This variant is used in some computational searches to minimize monochromatic cliques.

## Notes for Simulation/Search

- **Circulant structure**: Edge colors are invariant under vertex rotation (i → i+1 mod 43)
- **Symmetry**: Only need to compute ~n/2 = 21 difference values due to undirected edges
- **SIMD-friendly**: All vertex neighbors determined by O(1) modular arithmetic
- **K₅ verification**: Each of 43 guaranteed red K₅'s can be checked in O(1)
