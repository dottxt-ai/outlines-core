use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn fibonacci(n: u64) -> u64 {
    let mut a = 0;
    let mut b = 1;

    for _ in 0..n {
        let tmp = b;
        b += a;
        a = tmp;
    }

    a
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(15))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
