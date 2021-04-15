//! A supervised neural network model, which computes a forward pass, and updates parameters based on a target.

extern crate corgi;

use corgi::array::*;
use corgi::layers::dense::Dense;
use corgi::model::Model;
use corgi::nn_functions::{initializer, activation, cost};
use corgi::numbers::*;
use corgi::optimizers::gd::GradientDescent;

use criterion::{criterion_group, criterion_main, Criterion};

use rand::Rng;

fn dense(input_size: usize, output_size: usize, batch_size: usize, model: &mut Model) {
    let mut rng = rand::thread_rng();

    let mut input = vec![0.0; batch_size];
    let mut target = vec![0.0; batch_size];
    for j in 0..batch_size {
        let x: Float = rng.gen_range(-1.0..1.0);
        input[j] = x;
        target[j] = x.exp();
    }

    let input = Arrays::new((vec![batch_size, input_size], input));
    let target = Arrays::new((vec![batch_size, output_size], target));

    let _result = model.forward(input.clone());
    let _loss = model.backward(target.clone());
}

fn criterion_benchmark(c: &mut Criterion) {
    let learning_rate = 0.01;
    let batch_size = 16;
    let input_size = 1;
    let hidden_size = 2;
    let output_size = 1;
    let initializer = initializer::make_he();
    let sigmoid = activation::make_sigmoid();
    let mse = cost::make_mse();
    let gd = GradientDescent::new(learning_rate);
    let l1 = Dense::new(input_size, hidden_size, initializer.clone(), Some(sigmoid));
    let l2 = Dense::new(hidden_size, output_size, initializer.clone(), None);
    let mut model = Model::new(vec![Box::new(l1), Box::new(l2)], Box::new(gd), mse);

    c.bench_function("dense", move |b| {
        b.iter(|| dense(input_size, output_size, batch_size, &mut model))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
