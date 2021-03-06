//! A supervised neural network model, which computes a forward pass, and updates parameters based on a target.

extern crate corgi;

use corgi::array::*;
use corgi::layer::dense::Dense;
use corgi::model::Model;
use corgi::{initializer, activation, cost};
use corgi::numbers::*;
use corgi::optimizer::gd::GradientDescent;

use criterion::{criterion_group, criterion_main, Criterion};

use rand::Rng;

use std::time::Duration;

fn dense(input_size: usize, output_size: usize, batch_size: usize, model: &mut Model) {
    let mut rng = rand::thread_rng();

    let mut input = vec![0.0; batch_size];
    let mut target = vec![0.0; batch_size];
    for j in 0..batch_size {
        let x: Float = rng.gen_range(-1.0..1.0);
        input[j] = x;
        target[j] = x.exp();
    }

    let input = Array::from((vec![batch_size, input_size], input));
    let target = Array::from((vec![batch_size, output_size], target));

    let _result = model.forward(input.clone());
    let _loss = model.backward(target.clone());
    model.update();
}

fn criterion_benchmark(c: &mut Criterion) {
    let learning_rate = 0.01;
    let batch_size = 128;
    let input_size = 1;
    let hidden_size = 512;
    let output_size = 1;
    let initializer = initializer::he();
    let sigmoid = activation::sigmoid();
    let mse = cost::mse();
    let gd = GradientDescent::new(learning_rate);
    let mut l1 = Dense::new(input_size, hidden_size, &initializer, Some(&sigmoid));
    let mut l2 = Dense::new(hidden_size, output_size, &initializer, None);
    let mut model = Model::new(vec![&mut l1, &mut l2], &gd, &mse);

    let mut group = c.benchmark_group("Dense");
    group.measurement_time(Duration::from_secs(10));
    group.bench_function("dense", move |b| {
        b.iter(|| dense(input_size, output_size, batch_size, &mut model))
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
