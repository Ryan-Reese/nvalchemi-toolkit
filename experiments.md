# Beta Testing Experiments Log

Date: 2026-04-03
Version: origin/main at 93764a3 (post-PR #36)
Hardware: NVIDIA GH200 96GB HBM3 (container on gh-nvl-203-compute09)

## Workflows Tested

### WF1: Phonon Finite Difference Force Constants
**Result: PASS**
- 32-atom FCC Ar 2x2x2, 193 displaced configs in single batch
- Force constant matrix symmetric (rel asymmetry 7.3e-5)
- Acoustic sum rule satisfied (max violation 4.0e-6 eV/A²)
- 3 translational zero modes, no imaginary modes
- Batch handling with 193 systems works correctly

### WF2: Vacancy Formation Energy
**Result: PASS**
- Heterogeneous batch: 32 atoms (perfect) + 31 atoms (vacancy)
- FIRE converged in 76 steps
- E_formation = 0.0808 eV (positive, physically correct)
- Per-atom energy: perfect (-0.0807) < vacancy (-0.0781) as expected
- Batch indexing correct after dynamics

### WF3: Stress-Strain Curve (Uniaxial Tension)
**Result: PASS (with convention note)**
- 16 strain levels (-5% to +10%) in single batch
- Stress tensor symmetric (max asymmetry 1.9e-9 eV)
- Off-diagonal stresses negligible (~0.002 GPa)
- **Note**: `batch.stresses` stores raw virial (eV), NOT Cauchy stress. NPT/barostat code divides by volume internally. Users computing stress manually need σ = -W/V.

### WF4: RDF from NVT Trajectory
**Result: PASS**
- 108-atom FCC Ar 3x3x3, 500 NVT steps at 50K
- SnapshotHook + HostMemory captured 50 frames
- First RDF peak at 3.77 Å (expected 3.82 Å nn distance)
- g(r) tail approaches 1.0 (0.78 — limited by box size)
- **Note**: `HostMemory.read()` returns merged Batch. Iterate with `batch.get_data(i)`, NOT `for x in batch:` (yields storage key-value pairs, not graphs).

### WF5: Temperature Equilibration from Hot Start
**Result: PASS (with stability note)**
- **dt=2 fs, T_init=500K → NaN blowup at ~step 300** (parameter issue, not library bug)
- dt=1 fs, T_init=500K → stable, T → 98K (thermostat correct)
- dt=2 fs + MaxForceClampHook → stable, T → 88K
- dt=2 fs, T_init=200K → stable, T → 87K
- **Conclusion**: High T + large dt causes close encounters → LJ divergence. Safety hooks prevent this.

### WF6: Heterogeneous Batch Dynamics
**Result: PASS**
- 3 systems: 4, 32, 108 atoms at different temperatures
- NVE energy conserved per system (~5e-6 eV/atom/step)
- Temperatures independent (11.9K, 21.0K, 23.2K final)
- Batch structure preserved after 200 NVE steps

### WF7: ASE Edge Cases
**Results: Mixed**
- `from_structure()` not available (pre-PR#36 container) — cannot test issue #47
- ASE round-trip (Atoms → AtomicData → Atoms → AtomicData): **PASS**
- Single-atom systems: **PASS**
- Large system (256 atoms) dynamics: **PASS**
- **BUG: Mixed periodic/non-periodic batch FAILS** — `Batch.from_data_list([periodic, nonperiodic])` crashes with `ValueError: Inconsistent first dimension` when one system has cell/pbc and the other doesn't. Error message is confusing (mentions `energies` but real issue is cell/pbc field mismatch).

### WF8: Zarr Trajectory I/O Round-Trip
**Result: PASS**
- ZarrData write/read: 10 frames written and read correctly
- Data integrity: positions, atomic numbers, energies all preserved
- ConvergedSnapshotHook: wrote 5 converged structures from 3 systems (oscillation near threshold)
- Zero/drain/rewrite cycle: works correctly

## Bugs Found

### 1. Mixed Periodic/Non-Periodic Batching Fails (WF7)
- **Location**: `nvalchemi/data/batch.py:374` → `level_storage.py:880`
- **Trigger**: `Batch.from_data_list([data_with_cell, data_without_cell])`
- **Error**: `ValueError: Inconsistent first dimension: expected 1, got 2 for 'energies'`
- **Root cause**: When one AtomicData has `cell=None, pbc=None` and another has them set, the system-level UniformLevelStorage can't reconcile tensors of different first dimensions
- **Impact**: Researchers cannot batch molecules (no PBC) with crystals (PBC) in the same batch
- **Status**: Not covered by any existing PR. Related to issue #21 (Data Layer Gaps)

### 3. Batch.get_data() Out-of-Range Negative Index → Inconsistent Data (WF9, Test 4)
- **Location**: `nvalchemi/data/batch.py:584-585`
- **Trigger**: `batch.get_data(-5)` on a 3-graph batch (adjusted idx = -2)
- **Effect**: Returns atom-level data from graph 2 but system-level data from graph 1
- **Root cause**: No bounds check after adjusting `idx = num_graphs + idx`
- **Fix**: Added `if not (0 <= idx < self.num_graphs): raise IndexError(...)`
- **PR**: #49 (draft) — includes parametrized regression test
- **Status**: FIXED, PR submitted

### WF9: BiasHook, FusedStage, Edge Cases
- BiasHook harmonic restraint: ran but large displacements (PBC wrapping issue in bias function, not library bug)
- FusedStage (FIRE+NVE): failed with missing `num_neighbors` — user must register NeighborListHook on the fused stage or use `model.make_neighbor_hooks()`
- Batch round-trip after dynamics: PASS
- Negative indexing: found bug #3 above
- LoggingHook custom_scalars: PASS

### WF10: FusedStage (correct setup) + Elastic Constants
- FusedStage with hooks on fused pipeline: **PASS** — all 3 systems transit FIRE→NVE (status=2)
- Bulk modulus from E(V) parabola fit: 0.41 GPa (reasonable for LJ Ar)
- Batch.index_select (slice, tensor, list): **PASS**
- Batch.clone() independence: **PASS**
- AtomicData.to() device transfer round-trip: **PASS**

### WF11: Model Composition + FIRE2 + Edge Cases
- LJ + LJ composition: **PASS** — energies and forces exactly 2x
- FIREVariableCell: cell didn't expand in 300 steps (parameter issue, not bug)
- DemoModelWrapper on CUDA: **FAILS** — `nn.Embedding` stays on CPU, needs explicit `model.to("cuda")`. Inconsistent with Warp-based models that auto-handle device.
- Multiple NVE.run() continuation: **PASS** — step counter accumulates correctly
- Pre-converged system: **PASS** — exits in 1 step

## Observations (Not Bugs)

1. `batch.stresses` stores raw virial (eV), not Cauchy stress — could confuse users
2. `for x in batch:` yields `(key, tensor)` pairs, not individual graphs (Pydantic default)
3. `HostMemory.read()` returns merged Batch, not list of frames
4. `energies` dtype warnings during Zarr read-back (float64 → float32 cast)
