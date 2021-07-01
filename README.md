# mnist-digit-recognition-in-rust
Implementing handwritten digit recognition on the MNIST dataset using a feedfoward neural network using the rust programming language. 
Made for learning purposes. 

### Setup
1. Create a data folder for storage of the mnist data. 
2. Download the four files from this site: http://yann.lecun.com/exdb/mnist/
3. Move the train and test files to the data directory in your work-dir and unpack them. 


### Usage
Just run the program with 'cargo run' or make a release build for more speed with 'cargo build --release'.
ATM the program is only able to do basic training on a subset of the dataset. 

## Acknowledgements
This is a basic rewrite of the code found in this repository with some exceptions using rust: [https://github.com/ansonmiu0214/mnist-digit-recognition-feedforward]
