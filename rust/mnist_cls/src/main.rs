use anyhow::bail;
use tch::{vision::imagenet, Kind, Tensor};
use std::time::{Instant};

pub const CLASSES: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
pub const CLASS_COUNT: i64 = 10;

pub fn top_k(tensor: &Tensor, k: i64) -> Vec<(f64, String)> {
    let tensor = match tensor.size().as_slice() {
        [CLASS_COUNT] => tensor.shallow_clone(),
        [1, CLASS_COUNT] => tensor.view((CLASS_COUNT,)),
        [1, 1, CLASS_COUNT] => tensor.view((CLASS_COUNT,)),
        _ => panic!("unexpected tensor shape {tensor:?}"),
    };
    let (values, indexes) = tensor.topk(k, 0, true, true);
    let values = Vec::<f64>::try_from(values).unwrap();
    let indexes = Vec::<i64>::try_from(indexes).unwrap();
    values
        .iter()
        .zip(indexes.iter())
        .map(|(&value, &index)| (value, CLASSES[index as usize].to_owned()))
        .collect()
}

pub fn main() -> anyhow::Result<()> {
    
    let args: Vec<_> = std::env::args().collect();
    let (model_file, image_file) = match args.as_slice() {
        [_, m, i] => (m.to_owned(), i.to_owned()),
        _ => bail!("usage: main model.pt image.jpg"),
    };

    let start_load_image = Instant::now();
    let image = imagenet::load_image_and_resize224(image_file)?;
    let duration_load_image = start_load_image.elapsed();

    let start_load_model = Instant::now();
    let model = tch::CModule::load(model_file)?;
    let duration_load_model = start_load_model.elapsed();

    let start_forward = Instant::now();
    let output = model.forward_ts(&[image.unsqueeze(0)])?.softmax(-1, Kind::Float);
    let duration_forward = start_forward.elapsed();

    for (probability, class) in top_k(&output, 3).iter() {
        // let prob: f32 = probability.parse().unwrap();
        println!("{:5} {:5.2}%", class, 100.0 * probability);
        // println!("{:50} {:5.2}%", class, 100.0 * probability)
    }

    println!("duration_load_image is: {:?}", duration_load_image);
    println!("duration_load_model is: {:?}", duration_load_model);
    println!("duration_forward is: {:?}", duration_forward);

    Ok(())
}