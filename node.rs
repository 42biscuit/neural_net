pub type TreeIndex = usize;
#[derive(Clone, Debug)]
pub struct TreeNode {
    pub value: f32,
    pub next_nodes: Vec<Option<TreeIndex>>,
    pub weights: Vec<f32>,
    pub error: f32,
}

impl TreeNode {
    pub fn new(
        value: f32,
        next_nodes: Vec<Option<TreeIndex>>,
        weights: Vec<f32>,
        error: f32,
    ) -> Self {
        TreeNode {
            value: value,
            next_nodes: next_nodes,
            weights: weights,
            error: error,
        }
    }
    ///sigmoid squishificaiton function to keep the avtivation values and weights between 0 and 1
    pub fn squish(&mut self) {
        self.value = 1.0 / (1.0 + 2.71828_f32.powf(-self.value))
    }
}

impl ActivationFuncs<f64> for f64 {
    fn sigmoid_squish(&self) -> Option<f64> {
        Some(1.0 / (1.0 + 2.71828_f64.powf(-*self)))
    }
}
impl ActivationFuncs<f32> for f32 {
    fn sigmoid_squish(&self) -> Option<f32> {
        Some(1.0 / (1.0 + 2.71828_f32.powf(-*self)))
    }
}
pub trait ActivationFuncs<T> {
    fn sigmoid_squish(&self) -> Option<T>;
}

pub struct Stack<T> {
    inner: Vec<T>,
}
impl<T: std::cmp::PartialEq> Stack<T> {
    pub fn new() -> Self {
        Stack { inner: Vec::new() }
    }
    pub fn pop(&mut self) {
        if Some(&self.inner[0]) != None {
            self.inner.remove(self.inner.len() - 1);
        }
    }
    pub fn push(&mut self, data: T) {
        self.inner.push(data);
    }
}

pub trait Locational<T> {
    fn find(&self, key: T) -> Option<usize>;
}
impl Locational<Option<usize>> for Vec<Option<usize>> {
    fn find(&self, key: Option<usize>) -> Option<usize> {
        for (index, dat) in self.iter().enumerate() {
            if dat == &key {
                return Some(index);
            }
        }
        None
    }
}
impl Locational<usize> for Vec<usize> {
    fn find(&self, key: usize) -> Option<usize> {
        for (index, dat) in self.iter().enumerate() {
            if dat == &key {
                return Some(index);
            }
        }
        None
    }
}
