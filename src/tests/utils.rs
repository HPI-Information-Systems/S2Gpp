use std::net::SocketAddr;

use actix::{Actor, Addr, AsyncContext, Context, Handler, Message, System};
use actix_broker::BrokerSubscribe;
use actix_telepathy::prelude::*;
use log::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::utils::ClusterNodes;
use std::sync::{Arc, Mutex};

#[derive(RemoteMessage, Serialize, Deserialize)]
struct TestSortedMembersMessage(pub Vec<SocketAddr>);

#[derive(RemoteActor)]
#[remote_messages(TestSortedMembersMessage)]
pub struct TestClusterMemberListener {
    is_main: bool,
    main_socket_addr: SocketAddr,
    n_cluster_nodes: usize,
    local_host: SocketAddr,
    connected_nodes: HashSet<RemoteAddr>,
    main_node: Option<RemoteAddr>,
    pub(crate) sorted_nodes: HashMap<usize, RemoteAddr>,
    pub(crate) cluster_nodes: Arc<Mutex<Option<ClusterNodes>>>,
    sorted_addr_buffer: Vec<SocketAddr>,
}

impl TestClusterMemberListener {
    pub fn new(
        is_main: bool,
        main_socket_addr: SocketAddr,
        n_cluster_nodes: usize,
        local_host: SocketAddr,
        cluster_nodes: Arc<Mutex<Option<ClusterNodes>>>,
    ) -> Self {
        Self {
            is_main,
            main_socket_addr,
            n_cluster_nodes,
            local_host,
            connected_nodes: HashSet::new(),
            main_node: None,
            sorted_nodes: HashMap::new(),
            cluster_nodes,
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

            match remote_addr {
                Some(ra) => {
                    connected_nodes.remove(&ra);
                    self.sorted_nodes.insert(i, ra);
                }
                None => (),
            }
        }
    }

    fn finish_intro(&mut self) {
        *(self.cluster_nodes.lock().unwrap()) = Some(ClusterNodes::from(self.sorted_nodes.clone()));
    }
}

impl Actor for TestClusterMemberListener {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.subscribe_system_async::<ClusterLog>(ctx);
        self.register(ctx.address().recipient());
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        System::current().stop();
    }
}

impl Handler<ClusterLog> for TestClusterMemberListener {
    type Result = ();

    fn handle(&mut self, msg: ClusterLog, _ctx: &mut Self::Context) -> Self::Result {
        match msg {
            ClusterLog::NewMember(addr, remote_addr) => {
                debug!("new member {:?}", addr);

                if self.main_socket_addr.eq(&addr) {
                    self.main_node = Some(remote_addr.clone());
                }
                self.connected_nodes.insert(remote_addr);

                if self.connected_nodes.len() == self.n_cluster_nodes - 1 {
                    if self.is_main {
                        let mut sorted_members = vec![self.local_host.clone()];
                        sorted_members.append(
                            &mut self
                                .connected_nodes
                                .iter()
                                .map(|x| x.socket_addr.clone())
                                .collect(),
                        );

                        for node in self.connected_nodes.iter() {
                            let mut remote_listener = node.clone();
                            remote_listener.change_id("TestClusterMemberListener".to_string());
                            remote_listener
                                .do_send(TestSortedMembersMessage(sorted_members.clone()))
                        }

                        self.sort_members(sorted_members);
                        self.finish_intro();
                    } else if self.sorted_addr_buffer.len() > 0 {
                        self.sort_members(self.sorted_addr_buffer.clone());
                        self.finish_intro();
                    }
                }
            }
            ClusterLog::MemberLeft(addr) => {
                debug!("member left {:?}", addr);
            }
        }
    }
}

impl Handler<TestSortedMembersMessage> for TestClusterMemberListener {
    type Result = ();

    fn handle(&mut self, msg: TestSortedMembersMessage, _ctx: &mut Self::Context) -> Self::Result {
        if self.connected_nodes.len() == self.n_cluster_nodes - 1 {
            self.sort_members(msg.0);
            self.finish_intro();
        } else {
            self.sorted_addr_buffer = msg.0;
        }
    }
}

impl ClusterListener for TestClusterMemberListener {}
