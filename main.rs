#![allow(dead_code)]

mod node;
mod tree;

use node::{Locational, TreeNode};
use tree::{Tree, TreeBuilder};
fn main() {
    //  let tree_builder = TreeBuilder::new(4, 2, 0, 1);

    let mut tree = Tree::new(TreeBuilder::new(1, 1, 0, 0));

    for _ in 0..10 {
        println!("{:?}", tree.run_through_bf(&[0.5]));
        tree.backprop(&[0.9]);
    }

    let a = tree.add_node(TreeNode::new(9.0, vec![], vec![], 0.0));
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
    let mut input = 0.1;

    input = 1.0 / (1.0 + 2.71828_f64.powf(input));
    let mut weight = 0.1;
    let mut output;
    let target = 0.9;
    output = 1.0 / (1.0 + 2.71828_f64.powf(-(&input * &weight)));
    println!(
        "initial cost: {}    initial output:  {}",
        (&target - &output).powf(2.0),
        output
    );

    for _ in 0..6 {
        let delta_w = -150.0
            * (target - output)
            * (2.71828_f64.powf(-output) / ((1.0 + 2.71828_f64.powf(-output)).powf(2.0)))
            * input;
        weight = weight - delta_w;
        output = 1.0 / (1.0 + 2.71828_f64.powf(-(&input * &weight)));
        println!(
            "cost of adjusted network: {}     final output:   {}",
            (&target - &output).powf(2.0),
            output
        );
    }

    let a = vec![1 as usize, 2, 3, 4, 5];
    println!("{:?}\n ////////////////////////////////// \n  ", a.find(3));


    let builder = TreeBuilder::new(1, 2, 0,10 );
    let mut tree = Tree::new(builder);
    for _ in 0..100
     {
        println!("0.9 , 0.6 opts: {:?}",tree.run_through_bf(&[0.6,0.1]));
        tree.single_layer_bp(&[0.9,0.6]);
        println!("0.5 , 0.0 opts: {:?}",tree.run_through_bf(&[0.0,0.9]));
        tree.single_layer_bp(&[0.5,0.0]);
        // println!("0.5 opts: {:?}",tree.run_through_bf(&[0.1,0.9]));
        // tree.single_layer_bp(&[0.0,0.5]);

    }


}
