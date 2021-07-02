mod network;
mod io;
extern crate savefile;
use network::Network;
use savefile::prelude::*;
#[macro_use]
extern crate savefile_derive;

fn main() {
    let data = io::import_images();

    let layers = vec![784, 16, 10];

    let mut net: network::Network;
    let mut trained = false;

    match load_network() {
        Err(_) => net = network::Network::new(layers.clone()),
        Ok(saved_net) => {
            trained = true; 
            net = saved_net
        }
    }

    if !trained {
        train_network(&mut net, &data)
    }

    println!("Initialised network with layer structure {:?}", layers);

    save_network(&net);

    test_input_images(&net)
}

fn test_input_images(net: &Network) {
    let input = io::TestImages::new("data/test2.png");

    let guess = network::argmax(net.feedforward(&input.pixel_vector2d));

    println!("Pixel structure is: {:?}", input.pixel_vector2d);

    println!("The network guesses the number is: {}", guess)
}

fn train_network(net: &mut network::Network, data: &io::TrainData) {
    let mbs: u32 = 10;
    let learning_rate: f64 = 4.0;
    let success: u32 = 80;

    net.sgd(
        &data.training_data,
        &200, 
        success,
        &mbs, 
        &learning_rate, 
        &data.test_data
    );

    println!("Training successful!")
}

fn save_network(n: &network::Network) {
    save_file("data/network.bin", 0.1 as u32, n).unwrap();
    println!("Network saved. ");
}

fn load_network() -> Result<network::Network, SavefileError> {
    load_file("data/network.bin", 0.1 as u32)
}