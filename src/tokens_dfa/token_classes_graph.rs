use std::sync::Arc;
use std::cell::RefCell;
use rustc_hash::FxHashMap as HashMap;

use super::token_classes::{TokenClass, TokenClassId};




pub struct TokenClassNode {
    value: TokenClass,
    id: TokenClassId,
    leafs: HashMap<TokenClass, Arc<RefCell<TokenClassNode>>>,
}

pub struct TokenRootNode{
    first_node: Arc<RefCell<TokenClassNode>>
}

pub struct TokenClassesGraph {
    roots: HashMap<TokenClass, TokenRootNode>,
    class_ids: HashMap<TokenClass, TokenClassId>,
}

impl TokenClassNode {
    pub fn new(token_class: TokenClass, class_id: TokenClassId) -> Arc<RefCell<Self>> {
        Arc::new(RefCell::new(TokenClassNode {
            value: token_class,
            id: class_id,
            leafs: HashMap::default(),
        }))
    }

    pub fn get_value(&self) -> TokenClass { self.value.clone() }
    pub fn _get_class_id(&self) -> TokenClassId { self.id }
    pub fn _get_leafs(&self) -> Vec<Arc<RefCell<TokenClassNode>>> {
        self.leafs.values().cloned().collect()
    }

    pub fn add_leaf(&mut self, leaf: Arc<RefCell<TokenClassNode>>) -> bool {
       
        let leaf_value = leaf.borrow().get_value();
        if !leaf_value.starts_with(&self.value) {
            
            return false;
        }

        if self.leafs.contains_key(&leaf_value) {
            return true; // Nœud existe déjà
        }

        let mut leafs_to_remove = Vec::new();
        let mut found_next = None;

        for (existing_value, existing_leaf) in self.leafs.iter() {
            if leaf_value.starts_with(existing_value) {
                found_next = Some(existing_leaf.clone());
                break;
            }
            if existing_value.starts_with(&leaf_value) {
                leafs_to_remove.push(existing_value.clone());
            }
        }

        if !leafs_to_remove.is_empty() {
            
        
            // Cas d’inversion : déplacer les fils sous le nouveau nœud
            for key in &leafs_to_remove {
                let existing_leaf = self.leafs.remove(key).unwrap();
                leaf.borrow_mut().leafs.insert(key.clone(), existing_leaf);
            }
            self.leafs.insert(leaf_value.clone(), leaf);
            return true;
        } else if let Some(next_node) = found_next {
            // Parcours itératif vers le prochain niveau
            return next_node.borrow_mut().add_leaf(leaf);
        } else {
            // Insertion directe
            self.leafs.insert(leaf_value.clone(), leaf);
            return true;
        }

    }

    pub fn _print_tree(&self, depth: usize) {
        let indent = "  ".repeat(depth);
        println!("{}└─ id:{}-value:{:?}", indent, self.id, self.value);
        for leaf in &self.leafs {
            leaf.1.borrow()._print_tree(depth + 1);
        }
    }
}

impl TokenRootNode {
    
    pub fn new(first_node: Arc<RefCell<TokenClassNode>>)-> Self {
        TokenRootNode{
            first_node: first_node.clone()
        }
    }

    pub fn add_leaf(&mut self, leaf: Arc<RefCell<TokenClassNode>>) {
        
        if leaf.borrow().get_value().starts_with(&self.first_node.borrow().get_value()) {
            self.first_node.borrow_mut().add_leaf(leaf);   
        } else {
            let temp = self.first_node.clone();
            self.first_node = leaf.clone();
            self.first_node.borrow_mut().add_leaf(temp);
        }
    }

    pub fn get_first_node(&self) ->  Arc<RefCell<TokenClassNode>> {
        self.first_node.clone()
    }

    pub fn _get_prefix(&self) -> TokenClass {
        self.first_node.borrow().get_value()
    }

}

impl TokenClassesGraph {
    pub fn new() -> Self {
        TokenClassesGraph {
            roots: HashMap::default(),
            class_ids: HashMap::default(),
        }
    }

    pub fn add_class(&mut self, class: TokenClass, class_id: TokenClassId) {
        
        if self.class_ids.contains_key(&class) {
            return; // Ne rien faire si la classe est déjà dans le graphe
        }
        self.class_ids.insert(class.clone(), class_id);

        let node = TokenClassNode::new(class.clone(), class_id);

        let mut prefix:TokenClass = TokenClass::from_bytes(Vec::new());
        let mut find = false;
        for byte in class.as_bytes(){
            prefix.add_byte(*byte);
            if let Some(root) = self.roots.get_mut(&prefix) {
                find = true;
                root.add_leaf(node.clone());
                break;
            }
        }

        if !find {
            self.roots.insert(class, TokenRootNode::new(node.clone()));
            return;
        }
         
    }

    pub fn _print_tree(&self) {
        for node in self.roots.values() {
            node.get_first_node().borrow()._print_tree(0);
        }
    }
}




// Structure en lecture seule correspondante
pub struct ReadOnlyTokenClassNode {
    value: TokenClass,
    id: TokenClassId,
    leafs: HashMap<TokenClass, Arc<ReadOnlyTokenClassNode>>,
}

#[derive(Clone)]
pub struct ReadOnlyTokenRootNode {
    first_node: Arc<ReadOnlyTokenClassNode>
}

pub struct ReadOnlyTokenClassesGraph {
    roots: HashMap<TokenClass, ReadOnlyTokenRootNode>,

}

// Implémentation des méthodes
impl ReadOnlyTokenClassNode {
    pub fn get_value(&self) -> TokenClass { self.value.clone() }
    pub fn get_class_id(&self) -> TokenClassId { self.id }
    pub fn get_leafs(&self) -> Vec<Arc<ReadOnlyTokenClassNode>> {
        self.leafs.values().cloned().collect()
    }
    
    pub fn _print_tree(&self, depth: usize) {
        let indent = "  ".repeat(depth);
        println!("{}└─ id:{}-value:{:?}", indent, self.id, self.value);
        for leaf in &self.leafs {
            leaf.1._print_tree(depth + 1);
        }
    }
}

impl ReadOnlyTokenRootNode {
    pub fn get_first_node(&self) -> Arc<ReadOnlyTokenClassNode> {
        self.first_node.clone()
    }
    
    pub fn _get_prefix(&self) -> TokenClass {
        self.first_node.get_value()
    }
}

impl ReadOnlyTokenClassesGraph {
    pub fn get_iterator(&self) -> ReadOnlyTokenClassesGraphIterator {
        ReadOnlyTokenClassesGraphIterator::new(&self.roots
            .values()
            .map(|el| el.get_first_node())
            .collect()
        )
    }
    
    pub fn _print_tree(&self) {
        for node in self.roots.values() {
            node.get_first_node()._print_tree(0);
        }
    }

    pub fn get_roots(&self)->HashMap<TokenClass, ReadOnlyTokenRootNode> {
        self.roots.clone()
    }
}

// Fonction de conversion du graphe en version lecture seule
impl TokenClassesGraph {
    pub fn to_read_only(&self) -> ReadOnlyTokenClassesGraph {
        // Clonons d'abord les class_ids
        
        
        // Créons les racines en lecture seule
        let mut roots = HashMap::default();
        
        for (token_class, root_node) in &self.roots {
            // Convertir le noeud racine
            let read_only_first_node = Self::convert_node_to_read_only(&root_node.get_first_node().borrow());
            
            // Créer le noeud racine en lecture seule
            let read_only_root = ReadOnlyTokenRootNode {
                first_node: read_only_first_node
            };
            
            // Ajouter à la map des racines
            roots.insert(token_class.clone(), read_only_root);
        }
        
        ReadOnlyTokenClassesGraph {
            roots,
        }
    }
    
    fn convert_node_to_read_only(node: &TokenClassNode) -> Arc<ReadOnlyTokenClassNode> {
        // Convertir d'abord les feuilles
        let mut read_only_leafs = HashMap::default();
        
        for (token_class, leaf_node) in &node.leafs {
            let read_only_leaf = Self::convert_node_to_read_only(&leaf_node.borrow());
            read_only_leafs.insert(token_class.clone(), read_only_leaf);
        }
        
        // Créer le noeud en lecture seule
        Arc::new(ReadOnlyTokenClassNode {
            value: node.value.clone(),
            id: node.id,
            leafs: read_only_leafs,
        })
    }
}

// Implémentation de l'itérateur de lecture seule
pub struct ReadOnlyTokenClassesGraphIterator {
    nodes: Vec<Arc<ReadOnlyTokenClassNode>>,
    current: Option<Arc<ReadOnlyTokenClassNode>>
}

impl ReadOnlyTokenClassesGraphIterator {
    pub fn new(nodes: &Vec<Arc<ReadOnlyTokenClassNode>>) -> Self {
        ReadOnlyTokenClassesGraphIterator {
            nodes: nodes.clone(),
            current: None
        }
    }

    pub fn init(&mut self) -> Option<(TokenClass, TokenClassId)> {
        if self.nodes.is_empty() { return None; }
        self.current = self.nodes.pop();
        if let Some(current_ref) = &self.current {
            let class = current_ref.get_value();
            let id = current_ref.get_class_id();
            return Some((class, id));
        }
        None
    }

    pub fn has_child(&self) -> bool {
        if let Some(current) = &self.current {
            return current.get_leafs().len() > 0;
        }
        false
    }

    pub fn reject_and_advance(&mut self) -> Option<(TokenClass, TokenClassId)> {
        if self.nodes.is_empty() { return None; }
        self.current = self.nodes.pop();
        if let Some(current_ref) = &self.current {
            let class = current_ref.get_value();
            let id = current_ref.get_class_id();
            return Some((class, id));
        }
        None
    }

    pub fn accept_and_advance(&mut self) -> Option<(TokenClass, TokenClassId)> {
        if let Some(previous) = &self.current {
            let child = previous.get_leafs();
            self.nodes.extend(child.clone());
            self.current = self.nodes.pop();
            if let Some(current_ref) = &self.current {
                let class = current_ref.get_value();
                let id = current_ref.get_class_id();
                return Some((class, id));
            }
        }
        None
    }
    
    // Tu peux ajouter des méthodes similaires à TokenClassesGraphIterator
}