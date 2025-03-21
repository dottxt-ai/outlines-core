use std::io::Read;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use super::token_classes::{TokenClass, TokenClassId};

#[derive(Debug)]
pub struct ClassNode {
    id: TokenClassId,
    child: Vec<usize>
}

impl ClassNode {
    pub fn new(token_class_id:TokenClassId)->Self{
        ClassNode { id: token_class_id, child: Vec::new() }
    }

    pub fn add_children(&mut self, idx:usize){
        self.child.push(idx);
    }

    pub fn get_child(&self) -> &Vec<usize>{
        &self.child
    }

    pub fn get_class_id(&self) -> TokenClassId{
        self.id
    }
}
#[derive(Debug)]
pub struct PrefixGraph {
    root_class: Vec<u8>,
    nodes: Vec<ClassNode>
}

impl PrefixGraph {
    pub fn new(first_class:Vec<u8>, first_class_id:TokenClassId)-> Self{
        PrefixGraph{
            root_class: first_class,
            nodes: vec![ClassNode::new(first_class_id)],
        }
    }
    
    #[inline(always)]
    pub fn get_prefix(&self)-> &u8 {
        &self.root_class[0]
    }

    #[inline(always)]
    pub fn get_root_class(&self)-> &Vec<u8> {
        &self.root_class
    }

    #[inline(always)]
    pub fn get_root_class_id(&self) -> TokenClassId {
        self.nodes[0].get_class_id()
    }
    
    #[inline(always)]
    pub fn get_nodes_mut(&mut self)-> &mut Vec<ClassNode>{
        &mut self.nodes
    }
    
    #[inline(always)]
    pub fn get_nodes(&self) -> &Vec<ClassNode> {
        &self.nodes
    }

    #[inline(always)]
    pub fn iterator(&self) -> PrefixGraphIterator{
        PrefixGraphIterator::new(self)
    }

    pub fn print(&self) {
        println!("PrefixGraph (prefix: {:?}):", self.root_class);
        self.print_node(0, 0,  &mut HashSet::default());
    }

    fn print_node(&self, node_idx: usize, depth: usize,  visited: &mut HashSet<usize>) {
        
        if visited.contains(&node_idx) {
            return;
        }
        visited.insert(node_idx);

        let node = &self.nodes[node_idx];
        let indent = "  ".repeat(depth);
        
        println!("{}└─ Node[{}]", indent, node.id);
        
        for &child_idx in &node.child {
            self.print_node(child_idx, depth + 1,  visited);
        }
    }
}

pub struct PrefixGraphIterator<'a>{
    graph:&'a PrefixGraph,
    current_node:Option<&'a ClassNode>,
    stack_nodes: Vec<usize>
}

impl<'a> PrefixGraphIterator<'a>{
    pub fn new(graph: &'a PrefixGraph)->Self{
        PrefixGraphIterator { graph: graph, current_node: None, stack_nodes: vec![0] }
    }

    pub fn init(&mut self) {
        self.current_node = self.stack_nodes.pop().map(|node_id| &self.graph.get_nodes()[node_id]);
    }

    #[inline(always)]
    pub fn accept_and_advance(&mut self)  {     
        self.stack_nodes.extend(self.current_node.unwrap().child.iter());
        self.current_node = self.stack_nodes.pop().map(|node_id| &self.graph.get_nodes()[node_id]);
    }
    #[inline(always)]
    pub fn reject_and_advance(&mut self) {
        self.current_node = self.stack_nodes.pop().map(|node_id| &self.graph.get_nodes()[node_id]);
    }
    #[inline(always)]
    pub fn get_current(&self) -> Option<&'a ClassNode> {
        self.current_node
    }
}
#[derive(Debug)]
pub struct PrefixGraphes {
    graphes : Vec<PrefixGraph>,
    prefixes : HashMap<u8, HashSet<usize>> 

}

impl PrefixGraphes {
    
    pub fn new()-> Self {
        PrefixGraphes { graphes: vec![] , prefixes:HashMap::default()}
    }

    #[inline(always)]
    pub fn add_class(&mut self, class:&TokenClass, class_id:TokenClassId, classes: &Vec<TokenClass>){
       
        
        let mut find = false;

        if let Some(idxs) = self.prefixes.get(&class.prefix()){
            

            for idx in idxs {
                
                let graph = &mut self.graphes[*idx];
                if class.starts_with(&TokenClass::from_bytes(graph.get_root_class().to_vec())){
                    find = true;
                    let nodes_len = graph.get_nodes().len();
                    let nodes = graph.get_nodes_mut();
                    nodes.push(ClassNode::new(class_id));
                            
                    for node in nodes.iter_mut().rev().skip(1) {
                        if class.starts_with(&classes.as_slice()[node.id as usize]) {
                            
                            node.add_children(nodes_len);
                            break;
                        }
                    }
                    break;
                }
            }
            
        } 
        if !find {
            self.graphes.push(PrefixGraph::new(class.as_bytes().to_vec(), class_id));
            self.prefixes.entry(class.prefix()).or_default().insert(self.graphes.len()-1);
        }

    }

    pub fn get_graphes_from_prefix<'a>(&'a self, allowed_prefixes:&Vec<u8>, allowed_graphes:&mut Vec<&'a PrefixGraph>){
        allowed_graphes.clear();
        for allowed_prefix in allowed_prefixes{
        
            if let Some(idxs) = self.prefixes.get(allowed_prefix) {
                idxs.iter().for_each(|idx| {
                    allowed_graphes.push(&self.graphes[*idx]);
                });
                
            }
        }
    }

    pub fn print(&self) {
        println!("=== PrefixGraphes ===");
        println!("Total graphs: {}", self.graphes.len());
        
        // Afficher chaque graphe
        println!("\nGraphes:");
        for (i, graph) in self.graphes.iter().enumerate() {
            println!("\nGraph[{}]:", i);
            graph.print();
        }
    }
}


