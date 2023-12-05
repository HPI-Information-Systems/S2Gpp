use crate::parameters::{Parameters, Role};
use actix::{Actor, ActorContext, Addr, AsyncContext, Context, Handler, Message, System};
use actix_broker::BrokerSubscribe;
use actix_telepathy::{prelude::*, Node};
use log::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;

use crate::training::{StartTrainingMessage, Training};
use crate::utils::ClusterNodes;

#[derive(RemoteMessage, Serialize, Deserialize)]
struct SortedMembersMessage(pub Vec<SocketAddr>);

#[derive(RemoteActor)]
#[remote_messages(SortedMembersMessage)]
pub struct ClusterMemberListener {
    parameters: Parameters,
    connected_nodes: HashSet<Node>,
    main_node: Option<Node>,
    training: Addr<Training>,
    sorted_nodes: HashMap<usize, Node>,
    sorted_addr_buffer: Vec<SocketAddr>,
}

impl ClusterMemberListener {
    pub fn new(parameters: Parameters, training: Addr<Training>) -> Self {
        Self {
            parameters,
            connected_nodes: HashSet::new(),
            main_node: None,
            training,
            sorted_nodes: HashMap::new(),
            sorted_addr_buffer: vec![],
        }
    }

    fn sort_members(&mut self, sorted_socket_addrs: Vec<SocketAddr>) {
        let mut connected_nodes = self.connected_nodes.clone();
        for (i, socket_addr) in sorted_socket_addrs.into_iter().enumerate() {
            let remote_addr = connected_nodes.iter().find_map(|x| {
                if socket_addr.eq(&x.socket_addr) {
                    Some(x.clone())
                } else {
                    None
                }
            });

            if let Some(ra) = remote_addr {
                connected_nodes.remove(&ra);
                self.sorted_nodes.insert(i, ra);
            }
        }

        debug!("sorted: {:?}", self.sorted_nodes)
    }

    fn start_training(&mut self) {
        let nodes = ClusterNodes::from(self.sorted_nodes.clone());
        debug!("#nodes {}", nodes.len_incl_own());
        self.training.do_send(StartTrainingMessage {
            nodes,
            source: None,
            data: None,
        });
    }
}

impl Actor for ClusterMemberListener {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.subscribe_system_async::<ClusterLog>(ctx);
        self.register(ctx.address().recipient());
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        System::current().stop();
    }
}

impl Handler<ClusterLog> for ClusterMemberListener {
    type Result = ();

    fn handle(&mut self, msg: ClusterLog, ctx: &mut Self::Context) -> Self::Result {
        match msg {
            ClusterLog::NewMember(node) => {
                debug!("new member {:?}", node.socket_addr);

                if self.parameters.is_main_addr(node.socket_addr) {
                    debug!("is main node");
                    self.main_node = Some(node.clone());
                }
                self.connected_nodes.insert(node);

                if self.connected_nodes.len() == self.parameters.n_cluster_nodes - 1 {
                    match &self.parameters.role {
                        Role::Main { .. } => {
                            let mut sorted_members = vec![self.parameters.local_host];
                            sorted_members.append(
                                &mut self.connected_nodes.iter().map(|x| x.socket_addr).collect(),
                            );

                            for node in self.connected_nodes.iter() {
                                let remote_listener = node.get_remote_addr(Self::ACTOR_ID.to_string());
                                remote_listener
                                    .do_send(SortedMembersMessage(sorted_members.clone()))
                            }

                            self.sort_members(sorted_members);
                            self.start_training();
                        }
                        _ => {
                            if !self.sorted_addr_buffer.is_empty() {
                                self.sort_members(self.sorted_addr_buffer.clone());
                                self.start_training();
                            }
                        }
                    }
                }
            }
            ClusterLog::MemberLeft(addr) => {
                debug!("member left {:?}", addr);
                if let Role::Sub { mainhost: _ } = &self.parameters.role {
                    ctx.stop();
                }
            }
        }
    }
}

impl Handler<SortedMembersMessage> for ClusterMemberListener {
    type Result = ();

    fn handle(&mut self, msg: SortedMembersMessage, _ctx: &mut Self::Context) -> Self::Result {
        if self.connected_nodes.len() == self.parameters.n_cluster_nodes - 1 {
            debug!("received {:?}", msg.0);
            self.sort_members(msg.0);
            self.start_training();
        } else {
            self.sorted_addr_buffer = msg.0;
        }
    }
}

impl ClusterListener for ClusterMemberListener {}
