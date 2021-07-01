// Implements the image import and processing for use in the Neural Network
//#[macro_use]
//extern crate ndarray;
extern crate mnist;

use mnist::{Mnist, MnistBuilder};
use ndarray::{Array2};
use byteorder::{BigEndian, ReadBytesExt};
use std::convert::TryInto;

pub struct TrainData {
    train_array_2d: Vec<Vec<f64>>,
    train_label: Vec<u8>,
    pub training_data: Vec<Vec<Vec<f64>>>,
    validation_array_2d: Vec<Vec<f64>>,
    validation_label: Vec<u8>,
    pub validation_data: Vec<(Vec<f64>, i32)>,
    test_array_2d: Vec<Vec<f64>>,
    test_label: Vec<u8>,
    pub test_data: Vec<(Vec<f64>, i32)>,
}

pub fn import_images() -> TrainData {
    // Code was partly taken from this address (https://docs.rs/mnist/0.4.1/mnist/)
    // Except I didn't use matrix but ndarray instead
    let (trn_size, rows, cols) = (50_000, 28, 28);

    // Deconstruct the returned Mnist struct.
    let Mnist { trn_img, trn_lbl, 
                val_img, val_lbl, 
                tst_img, tst_lbl } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .base_path("data/")
        .finalize();
    
    let trn_array2d =  pack_images_vec(trn_img, rows, cols);
    let val_array2d = pack_images_vec(val_img, rows, cols);
    let tst_array2d = pack_images_vec(tst_img, rows, cols);
    
    TrainData {
        training_data: pack_training_data(&trn_array2d, &trn_lbl),
        validation_data: pack_vector_data(&val_array2d, &val_lbl),
        test_data: pack_vector_data(&tst_array2d, &tst_lbl),
        train_array_2d: trn_array2d,
        train_label: trn_lbl,
        validation_array_2d: val_array2d,
        validation_label: val_lbl,
        test_array_2d: tst_array2d,
        test_label: tst_lbl,
    }
}

fn normalize(x: &u8) -> f64 {
    // Designed to normalize Pixels without color-data so 255 is hardcoded
    *x as f64 / 255.0
}

fn pack_images(input_vector: Vec<u8>, rows: u8, cols: u8) -> Array2<f64> {
    // Transforms u8 values and pushes to out-Vector
    // Need to look up Ok/Error returns more

    let input: Vec<f64> = input_vector
        .iter()
        .map(normalize)
        .collect();

    let image_len = rows as usize * cols as usize;
    let arr_amount = input.len() / image_len;
    Array2::from_shape_vec((image_len, arr_amount), input)
        .unwrap()
}

fn pack_images_vec(input_vector: Vec<u8>, rows: u8, cols: u8) -> Vec<Vec<f64>> {
    let input: Vec<f64> = input_vector
        .iter()
        .map(normalize)
        .collect();
    
    let image_len = (rows as i32 * cols as i32) as usize;
    let vec_width = input.len() / image_len;
    let mut target_vec: Vec<Vec<f64>> = input.chunks(image_len)
        .map(|x| x.to_vec())    
        .collect();
    
    target_vec
}

fn pack_training_data(data: &Vec<Vec<f64>>, label: &Vec<u8>) -> Vec<Vec<Vec<f64>>> {
    let mut td: Vec<Vec<Vec<f64>>> = Vec::new();
    let num_vec: Vec<Vec<f64>> = label.iter().map(|x| vectorise_num(x)).collect();

    for (x, y) in data.iter().zip(&num_vec) {
        td.push(vec![x.clone(), y.clone()])
    }
    td
}

fn pack_vector_data(data: &Vec<Vec<f64>>, label: &Vec<u8>) -> Vec<(Vec<f64>, i32)>{
    let mut result: Vec<(Vec<f64>, i32)> = Vec::new();
    for (x, y) in data.iter().zip(label) {
        result.push((x.clone(), *y as i32))
    }
    result
}

fn vectorise_num(num: &u8) -> Vec<f64> {
    // Creates a 1d vector with size 10 (0..9) which contains a 1 a the 
    // index-location of the input @num

    let mut out = vec![0.; 10];
    out[*num as usize] = 1.;
    out
}