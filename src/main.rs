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

fn main() {
    let training_len: usize = 52_000;

    let Mnist {
        trn_img, trn_lbl, tst_img, tst_lbl, ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(training_len as u32)
        .validation_set_length(8_000)
        .test_set_length(10_000)
        .finalize();

    let learning_rate = 0.003;
    let batch_size = 32;
    let input_size = 784;
    let hidden_size = 256;
    let output_size = 10;

    let initializer = initializer::make_he();
    let relu = activation::make_relu();
    let softmax = activation::make_softmax();
    let cross_entropy = cost::make_cross_entropy();
    let gd = GradientDescent::new(learning_rate);

    let l1 = Dense::new(input_size, hidden_size, initializer.clone(), Some(relu.clone()));
    let l2 = Dense::new(hidden_size, output_size, initializer.clone(), Some(softmax.clone()));
    let mut model = Model::new(vec![Box::new(l1), Box::new(l2)], Box::new(gd), cross_entropy);

    // only one iteration here, but for more, should shuffle the training set
    let batch_count = training_len / batch_size;
    for i in 0..batch_count{
        let input = (0..batch_size * input_size).map(|j| trn_img[j + i * batch_size * input_size] as Float / 255.0).collect::<Vec<Float>>();
        let mut target = vec![0.0; batch_size * output_size];
        for j in 0..batch_size {
            target[j * output_size + trn_lbl[i * batch_size + j] as usize] = 1.0;
        }

        let input = Array::from((vec![batch_size, input_size], input));
        let target = Array::from((vec![batch_size, output_size], target));

        let _result = model.forward(input.clone());
        let loss = model.backward(target.clone());
        model.update();

        if i % 100 == 0 {
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

            println!("{} - loss: {}", i * 100 / batch_count, loss);
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
        let mut result = Array::matmul((&a, false), (&b, false), None);
        assert_eq!(result, arr![arr![10.0]]);

        result.backward(None);
        assert_eq!(b.gradient().to_owned().unwrap(), arr![arr![1.0], arr![2.0], arr![3.0]]);
        assert_eq!(a.gradient().to_owned().unwrap(), arr![arr![3.0, 2.0, 1.0]]);
    }
}
