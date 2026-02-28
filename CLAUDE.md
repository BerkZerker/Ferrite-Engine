# Ferrite Engine

Micro-voxel ray tracing engine in Rust.

## Build Commands

```bash
cargo build                    # Debug build
cargo build --release          # Release build
cargo test --workspace         # Run all tests
cargo clippy --workspace       # Lint
cargo fmt --check              # Format check
```

## Architecture

See `ARCHITECTURE.md` for full details.

### Crate Dependency Rules (ENFORCED)

- `ferrite-core` — zero GPU deps, depended on by everything
- `ferrite-voxel` — NO GPU dependencies. Must be testable without a GPU.
- `ferrite-world` — NO GPU dependencies
- `ferrite-physics` — NO GPU dependencies
- `ferrite-render` — the ONLY crate that imports `ash` / GPU code
- `ferrite-editor` — NO GPU dependencies (uses `egui`)
- `app` — thin binary, assembles plugins

**ferrite-voxel must NEVER depend on ferrite-render.** Data flows one way: render reads from voxel types.

### Key Conventions

- Edition 2024, resolver v2
- `glam` 0.30 for math (matches Bevy 0.18)
- All shared deps pinned in workspace `[workspace.dependencies]`
- No `unwrap()` in library crates — use `Result` or `expect()` with context
- `unwrap()` is acceptable in tests and examples
- Prefer `thiserror` for public error types
- `#[repr(C)]` or `#[repr(transparent)]` on any type uploaded to GPU
- Morton encoding via `ferrite_core::morton` for all voxel indexing

### Voxel Constants

- Chunk size: 32^3
- Voxel size: 4cm (parameterized via `VOXEL_SIZE_CM`)
- Voxel(0) = air
