use ndarray::prelude::*;
use ndarray_csv::{Array2Reader};
use csv::{ReaderBuilder, Trim};
use std::path::Path;
use std::fs::{File};
use std::str::FromStr;


pub fn read_data(file_path: &str, skip_columns: usize, skip_rows: usize) -> Array2<f32> {
    let file = File::open(file_path).unwrap();
    let mut reader = ReaderBuilder::new().has_headers(true).trim(Trim::All).from_reader(file);
    let dims = reader.headers().unwrap().len();

    println!("Dataset has {} columns", dims);

    let dataset: Array2<f32> = reader.deserialize_array2_dynamic().unwrap();
    dataset
}
