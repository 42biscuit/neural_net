
use crate::{node, tree};
#[allow(unused_assignments)]
use crate::node::{Locational, TreeIndex, TreeNode,ActivationFuncs};
use rand::prelude::*;
#[derive(Clone, Debug)]
pub struct TreeBuilder {
    inputs: usize,
    outputs: usize,
    hidden_layers: usize,
    inner_depth: usize,
}
impl TreeBuilder {
    pub fn new(inputs: usize, outputs: usize, hidden_layers: usize, inner_depth: usize) -> Self {
        TreeBuilder {
            inputs,
            outputs,
            hidden_layers,
            inner_depth,
        }
    }
}
#[derive(Clone, Debug)]
pub struct Tree {
    arena: Vec<Option<TreeNode>>,
    roots: Vec<Option<TreeIndex>>,
    pub opts: Vec<Option<TreeIndex>>,
    builder: TreeBuilder,
}
#[allow(dead_code)]
pub struct Trainer<T, A> {
    base: Tree,
    generations: usize,
    population: usize,
    rand_rate: f32,
    test: fn(&mut A, &[f32]) -> Vec<T>, //returns angle,cart a
    test_struct: A,
    test_fail: fn(&[f32]) -> bool,
}
impl<T, A: Clone> Trainer<T, A> {
    pub fn new(
        base: TreeBuilder,
        generations: usize,
        population: usize,
        rand_rate: f32,
        test: fn(&mut A, &[f32]) -> Vec<T>,
        test_struct: A,
        test_fail: fn(&[f32]) -> bool,
    ) -> Self {
        Trainer {
            base: Tree::new(base),
            generations,
            population,
            rand_rate,
            test,
            test_struct,
            test_fail,
        }
    }
}

impl Tree {
    ///consumes a TreeBuilder and returns a general net
    pub fn new(builder: TreeBuilder) -> Self {
        let mut a = Tree {
            arena: Vec::new(),
            roots: Vec::new(),
            opts: Vec::new(),
            builder: builder.clone(),
        };
        let mut next_nodes = Vec::new();
        let mut current_nodes = Vec::new();
        for index in 0..builder.outputs {
            next_nodes.push(Some(a.add_node(TreeNode::new(0.0, vec![], vec![], 0.0))));
            a.opts.push(Some(index));
        }
        for _ in 0..builder.hidden_layers {
            for _ in 0..builder.inner_depth {
                current_nodes.push(Some(a.add_node(TreeNode::new(
                    0.0,
                    next_nodes.clone(),
                    vec![0.5; next_nodes.len()],
                    0.0,
                ))));
            }
            next_nodes = current_nodes.clone();
            current_nodes = vec![];
        }

        for _ in 0..builder.inputs {
            current_nodes.push(Some(a.add_node(TreeNode::new(
                0.0,
                next_nodes.clone(),
                vec![0.5; next_nodes.len()],
                0.0,
            ))));
            a.add_root(*current_nodes.last().unwrap());
        }
        println!("roots: {:?}", a.roots);
        a
    }
    //used to clear error and value feilds of a net, only nececarry when using back prop or between run throughs on the same unaltered graph
    fn clear_vals(&mut self) {
        for i in 0..self.arena.len() {
            let mut node = self.node_at_mut(i).unwrap();
            node.value = 0.0;
            node.error = 0.0;
        }
    }
    pub fn run_through_bf(&mut self, inputs: &[f32]) -> Vec<f32> {
        self.clear_vals();
        let mut iter = self.iter();
        for i in 0..self.roots.len() {
            //sets inputs in the array to the given inputs
            let mut input = self.node_at_mut(self.roots[i].unwrap()).unwrap();
            input.value = *inputs.get(i).unwrap();
        }
        let mut opts = Vec::new();
        while let Some(res) = iter.next_layer_prp(self) {
            opts = res;
        }
        opts
    }

    pub fn run_through_df(&mut self, _inputs: &[f32]) -> Vec<f32> {
        for i in &self.arena {
            println!("next nodes:{:?}", i.as_ref().unwrap().next_nodes);
        }
        let mut opts = Vec::new();

        for _ in 0..self.roots.len() {
            let mut pos_iter = self.iter();
            while let Some(node) = pos_iter.next(&self) {
                let crnt_node = self.node_at(node).unwrap().clone();
                opts.push(crnt_node.value);
                for next_index in crnt_node.next_nodes.iter() {
                    let a = self.node_at_mut(next_index.unwrap()).unwrap();
                    opts.push(a.value);
                }
            }
        }
        opts
    }
    pub fn backprop(&mut self, expected_outputs: &[f32]) {
        let mut iter = self.iter_bk();

        for (index, expected) in self.opts.clone().iter().zip(expected_outputs.iter()) {
            let mut node = self.node_at_mut(index.unwrap()).unwrap();
            node.error = *expected;
            println!("opt node: {}  cost: {}", index.unwrap(), node.error);
        }
        while let Some(opts) = iter.next_layer_bk(self, 1.0) {}
    }
    pub fn single_layer_bp(&mut self, expected_outputs: &[f32]) {
        let mut tree_iter = self.iter_bk();                 //make backwards iterator
        for (index,nodpos) in self.opts.clone().iter().enumerate().rev(){             //populate ouptuts with expected values
            let mut node = self.node_at_mut(nodpos.unwrap()).unwrap();
            node.error = node.value - expected_outputs[index];
        }

        while let Some(nodes) = tree_iter.find_prev_layer(&self.clone()){
            tree_iter.update_W(self);
            tree_iter.update_errors(self);

            tree_iter.move_next_crnt();
            // println!("{:?}",nodes);
        }
    }


    pub fn bf_nodes_order(&mut self) -> Vec<TreeIndex> {
        let mut order = Vec::new();
        let mut arena_iter = self.iter();
        let mut Nweights = 0;
        while let Some(node) = arena_iter.next(&self) {
            order.push(node);
            if self.node_at(node).unwrap().weights.len() != Nweights {
                Nweights = self.node_at(node).unwrap().weights.len();
            }
        }
        order
    }

    fn evolve(&self, rand_rate: f32) -> Self {
        let mut tree = self.clone();
        let mut rng = rand::thread_rng();
        let mut itera = tree.iter();
        while let Some(index) = itera.next(&tree) {
            let node = tree.node_at_mut(index).unwrap();
            for i in 0..node.weights.len() {
                node.weights[i] += rng.gen_range(-1.0..1.0) * rand_rate;
            }
        }
        tree
    }
    pub fn iter(&self) -> PreorderIter {
        PreorderIter::new(self.roots.clone())
    }
    pub fn iter_bk(&self) -> PreorderIter {
        PreorderIter::newbk(self.opts.clone())
    }
    ///adds a root or "input" to the tree, from here iterators can be built to traverse the tree
    pub fn add_root(&mut self, root: Option<TreeIndex>) {
        self.roots.push(root);
    }

    pub fn add_node(&mut self, node: TreeNode) -> TreeIndex {
        let index = self.arena.len();
        self.arena.push(Some(node));
        return index;
    }

    pub fn _remove_node_at(&mut self, index: TreeIndex) -> Option<TreeNode> {
        if let Some(node) = self.arena.get_mut(index) {
            node.take()
        } else {
            None
        }
    }

    pub fn node_at(&self, index: TreeIndex) -> Option<&TreeNode> {
        return if let Some(node) = self.arena.get(index) {
            node.as_ref()
        } else {
            None
        };
    }

    pub fn node_at_mut(&mut self, index: TreeIndex) -> Option<&mut TreeNode> {
        return if let Some(node) = self.arena.get_mut(index) {
            node.as_mut()
        } else {
            None
        };
    }
} 

pub struct PreorderIter {
    next_nodes: Vec<Option<TreeIndex>>,
    crnt: Vec<Option<TreeIndex>>,
}

impl PreorderIter {
    pub fn new(roots: Vec<Option<TreeIndex>>) -> Self {
        PreorderIter {
            crnt: roots,
            next_nodes: vec![],
        }
    }
    pub fn newbk(opts: Vec<Option<TreeIndex>>) -> Self {
        PreorderIter {
            crnt: opts,
            next_nodes: vec![],
        }
    }
    fn move_next_crnt(&mut self){
        self.crnt = self.next_nodes.clone();
        self.next_nodes = vec![];
    }
    fn find_prev_layer(&mut self, tree: &Tree) -> Option<usize> {
        for index in 0..tree.arena.len() {
            if tree.node_at(index).unwrap().next_nodes == self.crnt {
                self.next_nodes.push(Some(index));
            }
        }
        if self.next_nodes.len() == 0 {
            None
        } else {
            Some(self.next_nodes.len())
        }
    }
    pub fn update_errors(&self, tree: &mut Tree) {
        for next_index in &self.next_nodes{
            let mut next_node = tree.node_at_mut(next_index.unwrap()).unwrap().clone().weights.clone();
            for (weight,crnt_index) in next_node.iter().rev().zip(self.crnt.iter()){
                tree.node_at_mut(next_index.unwrap()).unwrap().error += weight*&tree.node_at(crnt_index.unwrap()).unwrap().value;
            }
            let a = tree.node_at_mut(next_index.unwrap()).unwrap();
            a.squish();
            a.error = a.value - a.error;
        }


    }
    fn update_W(&self, tree: &mut Tree) {
        for next in &self.next_nodes {    //for each next node
            for crnt in &self.crnt {   //look at every current node
                let pos_crnt_self = &self.crnt.find(*crnt);  //find the location of the crnt in 
                let output = tree.node_at(crnt.unwrap()).unwrap().value.clone();
                let error = tree.node_at(crnt.unwrap()).unwrap().error.clone();
                tree.node_at_mut(
                    next.unwrap()).unwrap()               //get tree at index next_index
                .weights[pos_crnt_self.unwrap()] +=                                    //take weight from node at index next_index
                    50.0 * error.sigmoid_squish().unwrap()
                        * -(error)
                        * (2.71828_f32.powf(-output)
                            / ((1.0 + 2.71828_f32.powf(-output)).powf(2.0)))
                        * tree.node_at(pos_crnt_self.unwrap()).unwrap().value;
            }
        }
    }

    /*pub fn wb_errors(&mut self, tree: &mut Tree) -> Option<Vec<f32>> {
        //find node in previous layer
    }*/

    ///returns error values for the previous layer to allow bck prop
    pub fn next_layer_bk(&mut self, tree: &mut Tree, learning_rate: f32) -> Option<Vec<f32>> {
        //treat the next nodes feild as containing the nodes in the previous layer
        //initially the output nodes will have their error values loaded in
        let hold_index = self.crnt[0].unwrap();
        //find nodes in previous layer
        for i in 0..tree.arena.len() {
            if tree
                .node_at(i)
                .unwrap()
                .next_nodes
                .contains(&Some(hold_index))
            {
                self.next_nodes.push(Some(i));
            }
        }
        if self.next_nodes.len() == 0 {
            return None;
        }
        //do back prop here
        //next nodes contains previous layer
        //crnt contains the current layer
        //crnt nodes will have error pre loaded into them before each function call
        let mut iter = self.next_nodes.iter();
        let buff = tree.clone();
        while let Some(index) = iter.next() {
            let node = tree.node_at_mut(index.unwrap()).unwrap();
            for (weight, next) in node.weights.iter_mut().zip(node.next_nodes.iter()) {
                let activation = buff.node_at(next.unwrap()).unwrap().value;
                let delta_w = 10.0
                    * learning_rate
                    * -(&node.error - &node.value)
                    * (2.71828_f32.powf(-node.value)
                        / ((1.0 + 2.71828_f32.powf(-node.value)).powf(2.0)))
                    * activation;
                *weight -= delta_w;
            }
        }

        while let Some(index) = self.crnt.pop() {
            let activation = tree.node_at(index.unwrap()).unwrap().value;
            let weight_index = tree
                .node_at(self.next_nodes[0].unwrap())
                .unwrap()
                .next_nodes
                .binary_search(&index)
                .unwrap();
            for prev in 0..self.next_nodes.len() {
                let mut a = tree.node_at_mut(self.next_nodes[prev].unwrap()).unwrap();
                a.error += activation * a.weights[weight_index];
            }
        }
        let mut errors = Vec::new();
        for index in &self.next_nodes {
            let mut node = tree.node_at_mut(index.unwrap()).unwrap();
            node.error = 1.0 / (1.0 + 2.71828_f32.powf(-node.error)).powf(2.0);
            let temp = node.error - node.value;
            node.value = node.error;
            node.error = temp;
        }

        self.crnt = self.next_nodes.clone();
        self.next_nodes = vec![];
        if errors.len() == 9347230947520943652 {
            None
        } else {
            Some(errors)
        }
        //add previous nodes to next_nodes
    }

    ///takes the preorder iter and tree pair, returns a list of activations for the next layer of nodes

    pub fn next_layer_prp(&mut self, tree: &mut Tree) -> Option<Vec<f32>> {
        let mut activations = Vec::new();

        //next nodes feild is populated by all of the indexes of nodes in the connected nodes array of the first node in the current nodes array
        self.next_nodes = tree
            .node_at(self.crnt[0].unwrap())
            .unwrap()
            .next_nodes
            .clone();

        //iterates ofver the crnt nodes array in the preorderiter and
        while let Some(crnt_index) = self.crnt.pop() {
            let crnt_node = tree.node_at(crnt_index.unwrap()).unwrap().clone();

            for (next_index, weight) in crnt_node.next_nodes.iter().zip(crnt_node.weights.iter()) {
                let mut a = tree.node_at_mut(next_index.unwrap()).unwrap(); //a stores the value of the next node in the net, its activation is updated in this loop
                a.value += crnt_node.value * weight;
            }
        }
        for index in &self.next_nodes {
            let a = tree.node_at_mut(index.unwrap()).unwrap();
            a.squish();
            activations.push(a.value)
        }

        self.crnt = self.next_nodes.clone();

        if self.crnt.len() == 0 {
            None
        } else {
            Some(activations)
        }
    }

    pub fn next(&mut self, tree: &Tree) -> Option<TreeIndex> {
        while let Some(node_index) = self.crnt.pop() {
            if let Some(node) = tree.node_at(node_index.unwrap()) {
                for i in &*node.next_nodes {
                    self.crnt.push(*i);
                }

                return node_index;
            }
        }

        None
    } // immutable borrow &Tree ends here*/
}

#[cfg(test)]
mod tree_test {
    use crate::{
        node::TreeNode,
        tree::{Tree, TreeBuilder},
    };
    #[test]
    fn traversal_test() {
        let mut tree = Tree::new(TreeBuilder {
            inputs: 0,
            outputs: 0,
            hidden_layers: 0,
            inner_depth: 0,
        });
        let a = tree.add_node(TreeNode::new(4.0, vec![], vec![], 0.0));
        let b = tree.add_node(TreeNode::new(5.0, vec![], vec![], 0.0));
        let f = tree.add_node(TreeNode::new(6.0, vec![], vec![], 0.0));
        let d = tree.add_node(TreeNode::new(3.0, vec![], vec![], 0.0));
        let c = tree.add_node(TreeNode::new(
            2.0,
            vec![Some(a), Some(b), Some(f)],
            vec![],
            0.0,
        ));
        let e = tree.add_node(TreeNode::new(1.0, vec![Some(c), Some(d)], vec![], 0.0));
        tree.add_root(Some(e));
        let mut pos_iter = tree.iter();
        let mut values = Vec::new();
        while let Some(node) = pos_iter.next(&tree) {
            let node = tree.node_at(node).unwrap();
            values.push(node.value);
        }
        assert_eq!(values, vec![1.0, 3.0, 2.0, 6.0, 5.0, 4.0]);
    }
    #[test]
    fn depth_first_test() {
        let mut tree = Tree::new(TreeBuilder {
            inputs: 0,
            outputs: 0,
            hidden_layers: 0,
            inner_depth: 0,
        });
        let a = tree.add_node(TreeNode::new(4.0, vec![], vec![], 0.0));
        let b = tree.add_node(TreeNode::new(5.0, vec![], vec![], 0.0));
        let f = tree.add_node(TreeNode::new(6.0, vec![], vec![], 0.0));
        let d = tree.add_node(TreeNode::new(3.0, vec![], vec![], 0.0));
        let c = tree.add_node(TreeNode::new(
            2.0,
            vec![Some(a), Some(b), Some(f)],
            vec![],
            0.0,
        ));
        let _e = tree.add_node(TreeNode::new(1.0, vec![Some(c), Some(d)], vec![], 0.0));
        let order = tree.run_through_df(&[1.0, 1.0, 1.0, 1.0, 1.0]);
        println!("order {:?}", order);
    }
    #[test]
    fn tree_mut() {
        let mut tree = Tree::new(TreeBuilder {
            inputs: 0,
            outputs: 0,
            hidden_layers: 0,
            inner_depth: 0,
        });
        let a = tree.add_node(TreeNode::new(4.0, vec![], vec![], 0.0));
        let b = tree.add_node(TreeNode::new(5.0, vec![], vec![], 0.0));
        let f = tree.add_node(TreeNode::new(6.0, vec![], vec![], 0.0));
        let d = tree.add_node(TreeNode::new(3.0, vec![], vec![], 0.0));
        let c = tree.add_node(TreeNode::new(
            2.0,
            vec![Some(a), Some(b), Some(f)],
            vec![],
            0.0,
        ));
        let e = tree.add_node(TreeNode::new(1.0, vec![Some(c), Some(d)], vec![], 0.0));
        tree.add_root(Some(e));

        let mut pos_iter = tree.iter();
        while let Some(node) = pos_iter.next(&tree) {
            let node = tree.node_at_mut(node).unwrap();
            node.value *= 10.0;
        }

        let mut pos_iter = tree.iter();
        let mut values = Vec::new();
        while let Some(node) = pos_iter.next(&tree) {
            let node = tree.node_at(node).unwrap();
            values.push(node.value);
        }
        assert_eq!(values, vec![10.0, 30.0, 20.0, 60.0, 50.0, 40.0]);
    }
}

/*backprop alt (unfinished)
        let mut arena_index = Vec::new();
        let mut depths = Vec::new();
        depths.push(self.builder.inputs);
        for _ in 0..self.builder.hidden_layers{
            depths.push(self.builder.inner_depth);
        }
        depths.push(self.builder.outputs);
        for i in 0..(self.arena.len()-self.builder.inputs){
            arena_index.push(i);
        }
        for layer in (1..depths.len()).rev(){
            let mut pos = 0;
            for index in 0..depths[layer]{
                let crnt = self.node_at(arena_index[pos+index]).unwrap().value;
                for weight_index in 0..depths[layer-1]{
                    let weight = self.node_at_mut(pos+depths[layer]+weight_index).unwrap();
                    let deltaw = -(0.5 - crnt) * (2.71828_f32.powf(-crnt) / ((1.0 + 2.71828_f32.powf(-crnt)).powf(2.0))) * weight.value;
                    weight.weights
                }
            }
        pos += depths[layer];
        }

*/
