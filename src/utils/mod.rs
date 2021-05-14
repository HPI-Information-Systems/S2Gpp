use std::collections::HashMap;
use actix_telepathy::{RemoteAddr, AnyAddr};
use actix::{Recipient, Addr, Actor};
use std::any::Any;


#[derive(Default, Clone)]
pub struct ClusterNodes {
    nodes: HashMap<usize, RemoteAddr>
}

impl ClusterNodes {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn change_ids(&mut self, id: &str) {
        for (_, node) in self.nodes.iter_mut(){
            node.change_id(id.to_string());
        }
    }

    pub fn get_main_node(&self) -> Option<&RemoteAddr> {
        self.nodes.get(&0)
    }

    pub fn get(&self, key: &usize) -> Option<&RemoteAddr> {
        self.nodes.get(key)
    }

    pub fn to_any<T: Actor>(&self, addr: Addr<T>) -> AnyClusterNodes<T> {
        AnyClusterNodes::new(self.clone(), addr)
    }
}

impl From<HashMap<usize, RemoteAddr>> for ClusterNodes {
    fn from(nodes: HashMap<usize, RemoteAddr>) -> Self {
        Self {
            nodes,
            ..Default::default()
        }
    }
}

// AnyClusterNodes

pub struct AnyClusterNodes<T: Actor> {
    pub nodes: ClusterNodes,
    pub local_addr: Addr<T>
}

impl<T: Actor> AnyClusterNodes<T> {
    pub fn new(nodes: ClusterNodes, addr: Addr<T>) -> Self {
        Self {
            nodes,
            local_addr: addr
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.nodes.len()
    }
}

impl<T: Actor> IntoIterator for AnyClusterNodes<T> {
    type Item = AnyAddr<T>;
    type IntoIter = AnyClusterNodesIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        AnyClusterNodesIterator::from(self)
    }
}

impl<T: Actor> Clone for AnyClusterNodes<T> {
    fn clone(&self) -> Self {
        Self::new(self.nodes.clone(), self.local_addr.clone())
    }
}

// Iterator

pub struct AnyClusterNodesIterator<T: Actor> {
    any_cluster_nodes: AnyClusterNodes<T>,
    position: usize
}

impl<T: Actor> AnyClusterNodesIterator<T> {
    pub fn last_position(&mut self) -> bool {
        self.position >= self.any_cluster_nodes.len()
    }

    pub fn get_position(&mut self) -> usize {
        self.position
    }
}

impl<T: Actor> Iterator for AnyClusterNodesIterator<T> {
    type Item = AnyAddr<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.any_cluster_nodes.len() {
            None
        } else {
            let addr = match self.any_cluster_nodes.nodes.get(&self.position) {
                Some(node) => AnyAddr::Remote(node.clone()),
                None => AnyAddr::Local(self.any_cluster_nodes.local_addr.clone())
            };
            self.position += 1;
            return Some(addr)
        }
    }
}

impl<T: Actor> From<AnyClusterNodes<T>> for AnyClusterNodesIterator<T> {
    fn from(any_cluster_nodes: AnyClusterNodes<T>) -> Self {
        Self {
            any_cluster_nodes,
            position: 0
        }
    }
}
