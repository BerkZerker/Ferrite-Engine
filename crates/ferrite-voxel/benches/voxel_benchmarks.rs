use criterion::{Criterion, criterion_group, criterion_main};

use ferrite_core::coords::{CHUNK_VOLUME, LocalPos};
use ferrite_core::voxel::Voxel;
use ferrite_voxel::chunk::Chunk;
use ferrite_voxel::compression;
use ferrite_voxel::greedy_mesh::{self, ChunkNeighbors};
use ferrite_voxel::svo::Svo;
use ferrite_voxel::test_gen;

fn chunk_fill(c: &mut Criterion) {
    c.bench_function("chunk_fill_32768", |b| {
        b.iter(|| {
            let mut chunk = Chunk::new_air();
            for i in 0..CHUNK_VOLUME {
                let pos = LocalPos::from_index(i);
                chunk.set(pos, Voxel((i % 16 + 1) as u16));
            }
            chunk
        });
    });
}

fn chunk_get_all(c: &mut Criterion) {
    let mut chunk = Chunk::new_air();
    for i in 0..CHUNK_VOLUME {
        chunk.set(LocalPos::from_index(i), Voxel((i % 16 + 1) as u16));
    }

    c.bench_function("chunk_get_all_32768", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for i in 0..CHUNK_VOLUME {
                sum += chunk.get(LocalPos::from_index(i)).0 as u64;
            }
            sum
        });
    });
}

fn greedy_mesh_terrain(c: &mut Criterion) {
    let chunk = test_gen::generate_noise(42);
    let neighbors = ChunkNeighbors::none();

    c.bench_function("greedy_mesh_terrain", |b| {
        b.iter(|| greedy_mesh::greedy_mesh(&chunk, &neighbors));
    });
}

fn greedy_mesh_worst(c: &mut Criterion) {
    let chunk = test_gen::generate_checkerboard();
    let neighbors = ChunkNeighbors::none();

    c.bench_function("greedy_mesh_checkerboard", |b| {
        b.iter(|| greedy_mesh::greedy_mesh(&chunk, &neighbors));
    });
}

fn svo_build(c: &mut Criterion) {
    let chunk = test_gen::generate_noise(42);

    c.bench_function("svo_build_terrain", |b| {
        b.iter(|| Svo::build(&chunk));
    });
}

fn lz4_round_trip(c: &mut Criterion) {
    let chunk = test_gen::generate_noise(42);
    let compressed = compression::compress_chunk(&chunk);

    c.bench_function("lz4_compress", |b| {
        b.iter(|| compression::compress_chunk(&chunk));
    });

    c.bench_function("lz4_decompress", |b| {
        b.iter(|| compression::decompress_chunk(&compressed).unwrap());
    });
}

criterion_group!(
    benches,
    chunk_fill,
    chunk_get_all,
    greedy_mesh_terrain,
    greedy_mesh_worst,
    svo_build,
    lz4_round_trip,
);
criterion_main!(benches);
