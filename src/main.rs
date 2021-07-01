mod network;
mod io;
extern crate savefile;
use savefile::prelude::*;
#[macro_use]
extern crate savefile_derive;

fn main() {
    let data = io::import_images();
    println!("Data loaded. ");

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
    println!("Network saved. ")
}

fn train_network(net: &mut network::Network, data: &io::TrainData) {
    let mbs: u32 = 10;
    let learning_rate: f64 = 2.0;
    let success: u32 = 80;

    net.sgd(
        &data.training_data,
        &250, 
        success,
        &mbs, 
        &learning_rate, 
        &data.test_data
    );

    println!("Training successful!")
}

fn save_network(n: &network::Network) {
    save_file("data/network.bin", 0.1 as u32, n).unwrap()
}

fn load_network() -> Result<network::Network, SavefileError> {
    load_file("data/network.bin", 0.1 as u32)
}