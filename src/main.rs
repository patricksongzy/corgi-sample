//! A supervised neural network model, which computes a forward pass, and updates parameters based on a target.

#[cfg_attr(test, macro_use)]
extern crate corgi;

use corgi::array::*;
use corgi::layer::dense::Dense;
use corgi::{initializer, activation, cost};
use corgi::model::Model;
use corgi::numbers::*;
use corgi::optimizer::gd::GradientDescent;

use mnist::{Mnist, MnistBuilder};

use rand::thread_rng;
use rand::seq::SliceRandom;

fn main() {
    let training_len: usize = 52_000;
    let mut rand_indices: Vec<usize> = (0..52_000).collect();

    let Mnist {
        trn_img, trn_lbl, tst_img, tst_lbl, ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(training_len as u32)
        .validation_set_length(8_000)
        .test_set_length(10_000)
        .finalize();

    let learning_rate = 0.003;
    let iterations = 1;
    let batch_size = 32;
    let input_size = 784;
    let hidden_size = 256;
    let output_size = 10;

    let initializer = initializer::he();
    let relu = activation::relu();
    let softmax = activation::softmax();
    let cross_entropy = cost::cross_entropy();
    let gd = GradientDescent::new(learning_rate);

    let mut l1 = Dense::new(input_size, hidden_size, &initializer, Some(&relu));
    let mut l2 = Dense::new(hidden_size, output_size, &initializer, Some(&softmax));
    let mut model = Model::new(vec![&mut l1, &mut l2], &gd, &cross_entropy);

    let batch_count = training_len / batch_size;
    for _ in 0..iterations {
        rand_indices.shuffle(&mut thread_rng());
        for j in 0..batch_count {
            let input = (0..batch_size * input_size).map(|k| {
                trn_img[k + j * batch_size * input_size] as Float / 255.0
            }).collect::<Vec<Float>>();

            let mut target = vec![0.0; batch_size * output_size];
            for k in 0..batch_size {
                target[k * output_size + trn_lbl[j * batch_size + k] as usize] = 1.0;
            }

            let input = Array::from((vec![batch_size, input_size], input));
            let target = Array::from((vec![batch_size, output_size], target));

            let _result = model.forward(input.clone());
            let loss = model.backward(target.clone());
            model.update();

            if j % 100 == 0 {
                println!("{} - loss: {}", j * 100 / batch_count, loss);
            }
        }

        for j in 0..250 {
            let input = (0..input_size).map(|k| tst_img[k + j * input_size] as Float / 255.0).collect::<Vec<Float>>();
            let digit = tst_lbl[j];
            let input = Array::from((vec![input_size], input));
            let result = model.forward(input);
            let mut max = result[0];
            let mut max_index = 0;
            for k in 1..output_size {
                if result[k] > max {
                    max = result[k];
                    max_index = k;
                }
            }

            println!("{} - output: {}, target: {}", j, max_index, digit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array() {
        let a = arr![arr![1.0, 2.0, 3.0]].tracked();
        let b = arr![arr![3.0], arr![2.0], arr![1.0]].tracked();
        let result = Array::matmul((&a, false), (&b, false), None);
        assert_eq!(result, arr![arr![10.0]]);

        result.backward(None);
        assert_eq!(b.gradient().to_owned().unwrap(), arr![arr![1.0], arr![2.0], arr![3.0]]);
        assert_eq!(a.gradient().to_owned().unwrap(), arr![arr![3.0, 2.0, 1.0]]);
    }
}
