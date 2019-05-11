extern crate rand;

use rand::*;

use crate::cjsmat::*;
use crate::neural::*;

mod cjsmat;
mod neural;

fn main() {
    let mut neural_network_test: VecDenseNeuralNetwork<VecMutMatrix>
        = VecDenseNeuralNetwork::new(3, 2, 3, 2);

    neural_network_test.randomize_values(0.0, 1.0);
    println!("Neural network: {:#?}", neural_network_test);

    let mut rng = rand::thread_rng();
    let test_input = [rng.gen(), rng.gen(), rng.gen()];
    let test_output = neural_network_test.feed_forward(&test_input);
    println!("{:?}", test_output);
}
