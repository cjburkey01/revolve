use crate::cjsmat::*;

pub trait NeuralNetwork: Clone + std::fmt::Debug {
    fn get_input_nodes(&self) -> usize;
    fn get_output_nodes(&self) -> usize;
    fn feed_forward(&self, input_data: &[f64]) -> Box<[f64]>;
    fn randomize_values(&mut self, min: f64, max: f64);
}

pub trait DenseNeuralNetwork: NeuralNetwork {
    fn new(input_nodes: usize, hidden_layers: usize, nodes_per_hidden: usize, output_nodes: usize) -> Self;
    fn get_hidden_layers(&self) -> usize;
    fn get_nodes_per_hidden(&self) -> usize;
}

#[derive(Debug, Clone)]
pub struct VecDenseNeuralNetwork<T: MutMatrix> {
    input_nodes: usize,
    hidden_layers: usize,
    nodes_per_hidden: usize,
    output_nodes: usize,
    weight_matrices: Vec<T>,
}

impl<T: MutMatrix> VecDenseNeuralNetwork<T> {
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn sigmoid_vector(vector: &mut T) {
        let (rows, _) = vector.get_size();
        for row in 0..rows {
            vector.set_value(row, 0, Self::sigmoid(vector.get_value(row, 0)));
        }
    }

    fn add_mat_row(matrix: T, row_value: f64) -> T {
        matrix.add_row(&vec![row_value; matrix.get_cols()])
    }
}

impl<T: MutMatrix> NeuralNetwork for VecDenseNeuralNetwork<T> {
    fn get_input_nodes(&self) -> usize {
        self.input_nodes
    }

    fn get_output_nodes(&self) -> usize {
        self.output_nodes
    }

    fn feed_forward(&self, input_data: &[f64]) -> Box<[f64]> {
        if input_data.len() != self.input_nodes {
            panic!("Neural network expects same number of {} inputs, but {} were provided", self.input_nodes, input_data.len());
        }
        let mut moving_matrix = T::new_vec_from_slice(input_data);

        let weight_matrices = &self.weight_matrices;
        for i in 0..weight_matrices.len() {
            moving_matrix = weight_matrices[i].mul_mat(&Self::add_mat_row(moving_matrix, 1.0));
            Self::sigmoid_vector(&mut moving_matrix);
        }
        moving_matrix.get_column(0)
    }

    fn randomize_values(&mut self, min: f64, max: f64) {
        for weight_matrix in &mut self.weight_matrices {
            weight_matrix.randomize(min, max);
        }
    }
}

impl<T: MutMatrix> DenseNeuralNetwork for VecDenseNeuralNetwork<T> {
    fn new(input_nodes: usize, hidden_layers: usize, nodes_per_hidden: usize, output_nodes: usize) -> Self {
        if input_nodes == 0 || hidden_layers == 0 || nodes_per_hidden == 0 || output_nodes == 0 {
            panic!("A dense neural network must have at one input layer, hidden layer, node per hidden layer, and output layer, received {} {} {} and {}",
                   input_nodes, hidden_layers, nodes_per_hidden, output_nodes);
        }

        let weights = hidden_layers + 1;
        let mut network = Self {
            input_nodes,
            hidden_layers,
            nodes_per_hidden,
            output_nodes,
            weight_matrices: Vec::with_capacity(weights),
        };

        for layer in 0..weights {
            network.weight_matrices.push(T::new_value(if layer == weights - 1 { output_nodes } else { nodes_per_hidden },
                                                      (if layer == 0 { input_nodes } else { nodes_per_hidden }) + 1,
                                                      1.0));
        }

        network
    }

    fn get_hidden_layers(&self) -> usize {
        self.hidden_layers
    }

    fn get_nodes_per_hidden(&self) -> usize {
        self.nodes_per_hidden
    }
}
