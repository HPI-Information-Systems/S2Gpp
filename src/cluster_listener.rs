use std::net::SocketAddr;
use crate::parameters::{Parameters, Role};
use actix::{Actor, Context, System, Handler, ActorContext, AsyncContext, Addr, Message};
use actix_telepathy::prelude::*;
use actix_broker::{BrokerSubscribe};
use log::*;
use std::collections::{HashSet, HashMap};
use serde::{Serialize, Deserialize};


use crate::training::{Training, StartTrainingMessage};
use crate::utils::ClusterNodes;


#[derive(Message, RemoteMessage, Serialize, Deserialize)]
#[rtype(Result = "()")]
struct SortedMembersMessage(pub Vec<SocketAddr>);


#[derive(RemoteActor)]
#[remote_messages(SortedMembersMessage)]
pub struct ClusterMemberListener {
    parameters: Parameters,
    connected_nodes: HashSet<RemoteAddr>,
    main_node: Option<RemoteAddr>,
    training: Addr<Training>,
    sorted_nodes: HashMap<usize, RemoteAddr>,
    sorted_addr_buffer: Vec<SocketAddr>
}

impl ClusterMemberListener {
    pub fn new(parameters: Parameters, training: Addr<Training>) -> Self {
        Self {
            parameters,
            connected_nodes: HashSet::new(),
            main_node: None,
            training,
            sorted_nodes: HashMap::new(),
            sorted_addr_buffer: vec![]
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

            match remote_addr {
                Some(ra) => {
                    connected_nodes.remove(&ra);
                    self.sorted_nodes.insert(i, ra);
                },
                None => ()
            }
        }

        debug!("sorted: {:?}", self.sorted_nodes)
    }

    fn start_training(&mut self) {
        let nodes = ClusterNodes::from(self.sorted_nodes.clone());
        self.training.do_send(StartTrainingMessage { nodes });
    }
}

impl Actor for ClusterMemberListener {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.subscribe_system_async::<ClusterLog>(ctx);
        self.register(ctx.address().recipient(), "ClusterMemberListener".to_string());
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        System::current().stop();
    }
}

impl Handler<ClusterLog> for ClusterMemberListener {
    type Result = ();

    fn handle(&mut self, msg: ClusterLog, ctx: &mut Self::Context) -> Self::Result {
        match msg {
            ClusterLog::NewMember(addr, remote_addr) => {
                debug!("new member {:?}", addr);

                if self.parameters.is_main_addr(addr) {
                    self.main_node = Some(remote_addr.clone());
                }
                self.connected_nodes.insert(remote_addr);

                if self.connected_nodes.len() == self.parameters.n_cluster_nodes - 1 {
                    match &self.parameters.role {
                        Role::Main { .. } => {
                            let mut sorted_members = vec![self.parameters.local_host.clone()];
                            sorted_members.append(&mut self.connected_nodes.iter().map(|x| x.socket_addr.clone()).collect());

                            for node in self.connected_nodes.iter() {
                                let mut remote_listener = node.clone();
                                remote_listener.change_id("ClusterMemberListener".to_string());
                                remote_listener.do_send(SortedMembersMessage(sorted_members.clone()))
                            }

                            self.sort_members(sorted_members);
                            self.start_training();
                        },
                        _ => if self.sorted_addr_buffer.len() > 0 {
                            self.sort_members(self.sorted_addr_buffer.clone());
                            self.start_training();
                        }
                    }
                }
            },
            ClusterLog::MemberLeft(addr) => {
                debug!("member left {:?}", addr);
                match &self.parameters.role {
                    Role::Sub { mainhost } => {
                        if addr.eq(mainhost) {
                            ctx.stop();
                        }
                    },
                    _ => ()
                }
            }
        }
    }
}

impl Handler<SortedMembersMessage> for ClusterMemberListener {
    type Result = ();

    fn handle(&mut self, msg: SortedMembersMessage, _ctx: &mut Self::Context) -> Self::Result {
        if self.connected_nodes.len() == self.n_cluster_nodes - 1 {
            self.sort_members(msg.0);
            self.finish_intro();
        } else {
            self.sorted_addr_buffer = msg.0;
        }
    }
}

impl ClusterListener for ClusterMemberListener {}
