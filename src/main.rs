mod network;
mod io;

fn main() {
    let data = io::import_images();
    println!("Data loaded. ");

    let layers = vec![784, 16, 10];

    let mut net = network::Network::new(layers.clone());
    println!("Initialised network with layer structure {:?}", layers);

    let mbs: u32 = 10;
    let learning_rate: f64 = 2.0;
    net.sgd(
        &data.training_data,
        &66, 
        &mbs, 
        &learning_rate, 
        &data.test_data
    );
    println!("Training complete! ")



    

    // net.backpropagate(&vec![1.; 784], &vec![1.; 10]);

    //println!("Biases are: {:?}", net.biases);
    //println!("Weights are: {:?}", &net.weights.len());
    //println!("Dimensions of weights are: x:{:?}, y:{:?}", &net.weights[0].len(), &net.weights[1].len());
    //println!("Dimensions of subweights are: xx:{:?}", &net.weights[0][0].len());

}
