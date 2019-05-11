use rand::*;

pub trait Matrix: Clone + std::fmt::Debug {
    /// Creates a new fixed-size matrix with the provided dimensions.
    /// 
    /// # Parameters
    /// * `rows` - The number of rows this matrix will have
    /// * `cols` - The number of columns this matrix will have
    fn new(rows: usize, cols: usize) -> Self;

    /// Creates a new fixed-size matrix with the provided dimensions and all
    /// cells initialized to the provided value.
    /// 
    /// # Parameters
    /// * `rows` - The number of rows this matrix will have
    /// * `cols` - The number of columns this matrix will have
    /// * `value` - The default value for all cells
    fn new_value(rows: usize, cols: usize, value: f64) -> Self;

    /// Generate a new fix-size matrix with the provided dimensions that is
    /// initialized with uniformly random values between the provided minimum
    /// and maximum values.
    /// 
    /// # Parameters
    /// * `rows` - The number of rows this matrix will have
    /// * `cols` - The number of columns this matrix will have
    /// * `min` - The smallest possible random floating point
    /// * `max` - The largest possible random floating point
    fn new_random(rows: usize, cols: usize, min: f64, max: f64) -> Self;

    /// Creates a new column matrix (a vector) from the provided slice data.
    /// 
    /// # Parameters
    /// * `slice` - The slice from which to create a new vector
    fn new_vec_from_slice(slice: &[f64]) -> Self;

    /// Creates a new column matrix (a vector) from the provided slice data.
    /// 
    /// # Parameters
    /// * `slice` - The slice from which to create a new vector
    fn new_vec_from_slice_sized(slice: &[f64], size: usize) -> Self;

    /// Removes the given number of columns from the matrix.
    /// 
    /// # Parameters
    /// * `to_remove` - The number of columns to be removed from the matrix
    fn sub_col_matrix(&self, to_remove: usize) -> Self;

    /// Removes the given number of rows from the matrix.
    /// 
    /// # Parameters
    /// * `to_remove` - The number of rows to be removed from the matrix
    fn sub_row_matrix(&self, to_remove: usize) -> Self;

    /// Returns the number of rows and columns this matrix has.
    /// 
    /// # Returns
    /// The returned tuple has:
    /// * `usize` - The number of rows this matrix has
    /// * `usize` - The number of columns this matrix has
    fn get_size(&self) -> (usize, usize);

    /// Returns the number of rows this matrix has.
    #[inline]
    fn get_rows(&self) -> usize {
        let (rows, _) = self.get_size();
        rows
    }

    /// Returns the number of columns this matrix has.
    #[inline]
    fn get_cols(&self) -> usize {
        let (_, cols) = self.get_size();
        cols
    }

    /// Returns a slice containing the values of the given row of the matrix.
    /// 
    /// # Parameters
    /// * `row` - The row to retrieve
    fn get_row(&self, row: usize) -> Box<[f64]>;

    /// Returns a slice containing the values of the given column of the
    /// matrix.
    /// 
    /// # Parameters
    /// * `col` - The column to retrieve
    fn get_column(&self, col: usize) -> Box<[f64]>;

    /// Retrieves the current value, an `f64` from the given cell within the
    /// matrix.
    /// 
    /// # Parameters
    /// * `row` - The row from which the value should be retrieved
    /// * `col` - The column from which the value should be retrieved
    fn get_value(&self, row: usize, col: usize) -> f64;

    /// Adds the given slice as a row to the matrix and returns the new
    /// matrix.
    /// 
    /// # Parameters
    /// * `slice` - The new row to be added
    fn add_row(&self, slice: &[f64]) -> Self;

    /// Adds the given slice as a column to the matrix and returns the new
    /// matrix.
    /// 
    /// # Parameters
    /// * `slice` - The new column to be added
    fn add_column(&self, slice: &[f64]) -> Self;

    /// Transposes this matrix and returns the result.
    fn transpose(&self) -> Self;

    /// Multiplies every value within this matrix by the provided value and
    /// returns the results in a new matrix.
    /// 
    /// # Parameters
    /// * `value` - The 64-bit float value by which to multiple all values
    /// within the matrix
    fn mul_scalar(&self, value: f64) -> Self;

    /// Multiplies this matrix by the provided matrix and returns the new
    /// resulting matrix.
    /// 
    /// # Parameters
    /// * `value` - The multiplier matrix which must have the same number of
    /// rows as this matrix has columns
    fn mul_mat(&self, value: &Self) -> Self;

    /// Sums every value within this matrix by the provided value and returns
    /// the results in a new matrix.
    /// 
    /// # Parameters
    /// * `value` - The 64-bit float value to be added to the values in this
    /// matrix
    fn add_scalar(&self, value: f64) -> Self;

    /// Adds the provided matrix's values to this matrix's values and returns
    /// the results in a new matrix.
    /// 
    /// # Parameters
    /// * `value` - A matrix with the same rows and columns as this matrix
    fn add_mat(&self, value: &Self) -> Self;
}

pub trait MutMatrix: Matrix {
    /// Assigns the given cell to the provided value.
    /// 
    /// # Parameters
    /// * `row` - The row of the cell to be modified
    /// * `col` - The column of the cell to be modified
    /// * `value` - The new 64-bit float value of the given cell
    fn set_value(&mut self, row: usize, col: usize, value: f64);

    /// Mutably multiplies every value within this matrix by the provided value.
    /// This modifies this matrix's values.
    /// 
    /// # Parameters
    /// `value` - The 64-bit float value by which to multiple all values
    fn mul_scalar(&mut self, value: f64);

    /// Mutably sums every value within this matrix by the provided value.
    /// 
    /// # Parameters
    /// * `value` - The 64-bit float value to be added to the values in this
    /// matrix
    fn add_scalar(&mut self, value: f64);

    /// Mutably adds the provided matrix's values to this matrix's values.
    /// 
    /// # Parameters
    /// * `value` - A matrix with the same rows and columns as this matrix
    fn add_mat(&mut self, value: &Self);

    /// Assign all cells in this matrix to uniformly generated random values
    /// between the provided minimum and maximum.
    /// 
    /// # Parameters
    /// * `min` - The smallest possible random floating point
    /// * `max` - The largest possible random floating point
    fn randomize(&mut self, min: f64, max: f64);
}

#[derive(Debug, Clone)]
pub struct VecMutMatrix {
    rows: usize,
    cols: usize,
    mat: Vec<f64>,
}

impl VecMutMatrix {
    fn mul_scalar_raw(matrix: &mut Vec<f64>, value: f64) {
        for i in 0..matrix.len() {
            matrix[i] *= value;
        }
    }

    fn add_scalar_raw(matrix: &mut Vec<f64>, value: f64) {
        for i in 0..matrix.len() {
            matrix[i] += value;
        }
    }

    fn add_mat_raw(self_rows: usize, self_cols: usize, matrix: &mut Vec<f64>, other_rows: usize, other_cols: usize, value: &Vec<f64>) {
        if self_rows != other_rows || self_cols != other_cols {
            panic!("Two matrices being added together must have the same size, attempted to add a {},{} matrix with a {}, {}",
                   self_rows, self_cols, other_rows, other_cols);
        }
        for i in 0..matrix.len() {
            matrix[i] += value[i];
        }
    }

    fn randomize(size: usize, min: f64, max: f64, matrix: &mut Vec<f64>) {
        let mut rng = rand::thread_rng();
        for i in 0..size {
            matrix[i] = rng.gen_range(min, max);
        }
    }
}

impl Matrix for VecMutMatrix {
    fn new(rows: usize, cols: usize) -> Self {
        Self::new_value(rows, cols, 0.0)
    }

    fn new_value(rows: usize, cols: usize, value: f64) -> Self {
        Self { rows, cols, mat: vec![value; rows * cols] }
    }

    fn new_random(rows: usize, cols: usize, min: f64, max: f64) -> Self {
        let mut new_matrix = Self { rows, cols, mat: Vec::with_capacity(rows * cols) };
        Self::randomize(rows * cols, min, max, &mut new_matrix.mat);
        new_matrix
    }

    fn new_vec_from_slice(slice: &[f64]) -> Self {
        let mut new_matrix = Self::new(slice.len(), 1);
        for i in 0..slice.len() {
            new_matrix.set_value(i, 0, slice[i]);
        }
        new_matrix
    }

    fn new_vec_from_slice_sized(slice: &[f64], size: usize) -> Self {
        if size < slice.len() {
            panic!("New vector must have a size of at least as large as initial slice");
        }
        let mut new_matrix = Self::new(size, 1);
        for i in 0..slice.len() {
            new_matrix.set_value(i, 0, slice[i]);
        }
        new_matrix
    }

    fn sub_col_matrix(&self, to_remove: usize) -> Self {
        if to_remove == 0 {
            return self.clone();
        }
        if to_remove >= self.cols {
            panic!("Removing {} columns from matrix with {} columns would result in empty or negative matrix", to_remove, self.cols);
        }
        let mut new_matrix = Self::new(self.rows, self.cols - to_remove);
        for row in 0..new_matrix.rows {
            for col in 0..new_matrix.cols {
                new_matrix.set_value(row, col, self.get_value(row, col));
            }
        }
        new_matrix
    }

    fn sub_row_matrix(&self, to_remove: usize) -> Self {
        if to_remove == 0 {
            return self.clone();
        }
        if to_remove >= self.rows {
            panic!("Removing {} rows from matrix with {} rows would result in empty or negative matrix", to_remove, self.rows);
        }
        let mut new_matrix = Self::new(self.rows - to_remove, self.cols);
        for row in 0..new_matrix.rows {
            for col in 0..new_matrix.cols {
                new_matrix.set_value(row, col, self.get_value(row, col));
            }
        }
        new_matrix
    }

    fn get_size(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    fn get_row(&self, row: usize) -> Box<[f64]> {
        if row >= self.rows {
            panic!("Cannot retrieve row {} from matrix with only {} rows", row, self.rows);
        }
        let mut output = Vec::with_capacity(self.cols);
        for i in 0..self.cols {
            output.push(self.get_value(row, i));
        }
        output.into_boxed_slice()
    }

    fn get_column(&self, col: usize) -> Box<[f64]> {
        if col >= self.cols {
            panic!("Cannot retrieve column {} from matrix with only {} columns", col, self.cols);
        }
        let mut output = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            output.push(self.get_value(i, col));
        }
        output.into_boxed_slice()
    }

    fn get_value(&self, row: usize, col: usize) -> f64 {
        if row >= self.rows || col >= self.cols {
            panic!("Invalid matrix row or column: {},{} as matrix has size of {} rows by {} columns", row, col, self.rows, self.cols);
        }
        self.mat[row * self.cols + col]
    }

    fn add_row(&self, slice: &[f64]) -> Self {
        if slice.len() != self.cols {
            panic!("The matrix has {} columns but the row to be added has {}", self.cols, slice.len());
        }
        let mut new_matrix = Self::new(self.rows + 1, self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                new_matrix.set_value(row, col, self.get_value(row, col));
            }
        }
        for col in 0..new_matrix.cols {
            new_matrix.set_value(self.rows, col, slice[col]);
        }
        new_matrix
    }

    fn add_column(&self, slice: &[f64]) -> Self {
        if slice.len() != self.rows {
            panic!("The matrix has {} rows but the column to be added has {}", self.rows, slice.len());
        }
        let mut new_matrix = Self::new(self.rows, self.cols + 1);
        for row in 0..self.rows {
            for col in 0..self.cols {
                new_matrix.set_value(row, col, self.get_value(row, col));
            }
        }
        for row in 0..new_matrix.rows {
            new_matrix.set_value(row, self.cols, slice[row]);
        }
        new_matrix
    }

    fn transpose(&self) -> Self {
        let mut new_matrix = Self::new(self.cols, self.rows);
        for row in 0..self.rows {
            for col in 0..self.cols {
                new_matrix.set_value(col, row, self.get_value(row, col));
            }
        }
        new_matrix
    }

    fn mul_scalar(&self, value: f64) -> Self {
        let mut new_matrix = self.clone();
        Self::mul_scalar_raw(&mut new_matrix.mat, value);
        new_matrix
    }

    fn mul_mat(&self, value: &Self) -> Self {
        if self.cols != value.rows {
            panic!("Multiplicand matrix of matrix multiplication must have same number of columns as the multiplier has rows, given {} and {}",
                   self.cols, value.rows);
        }

        let mut new_matrix = Self::new(self.rows, value.cols);
        for row in 0..self.rows {
            for col in 0..value.cols {
                let mut cell_value = 0.0f64;
                for current_cell in 0..self.cols {
                    cell_value += self.get_value(row, current_cell) * value.get_value(current_cell, col);
                }
                new_matrix.set_value(row, col, cell_value);
            }
        }
        new_matrix
    }

    fn add_scalar(&self, value: f64) -> Self {
        let mut new_matrix = self.clone();
        Self::add_scalar_raw(&mut new_matrix.mat, value);
        new_matrix
    }

    fn add_mat(&self, value: &Self) -> Self {
        let mut new_matrix = self.clone();
        Self::add_mat_raw(self.rows, self.cols, &mut new_matrix.mat,
                          value.rows, value.cols, &value.mat);
        new_matrix
    }
}

impl MutMatrix for VecMutMatrix {
    fn set_value(&mut self, row: usize, col: usize, value: f64) {
        if row >= self.rows || col >= self.cols {
            panic!("Invalid matrix row or column: {},{} as matrix has size of {} rows by {} columns", row, col, self.rows, self.cols);
        }
        self.mat[row * self.cols + col] = value;
    }

    fn mul_scalar(&mut self, value: f64) {
        Self::mul_scalar_raw(&mut self.mat, value);
    }

    fn add_scalar(&mut self, value: f64) {
        Self::add_scalar_raw(&mut self.mat, value);
    }

    fn add_mat(&mut self, value: &Self) {
        Self::add_mat_raw(self.rows, self.cols, &mut self.mat,
                          value.rows, value.cols, &value.mat);
    }

    fn randomize(&mut self, min: f64, max: f64) {
        Self::randomize(self.rows * self.cols, min, max, &mut self.mat);
    }
}
