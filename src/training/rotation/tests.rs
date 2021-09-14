use std::sync::{Arc, Mutex};

use actix::{System};
use ndarray::{arr3, Array3};
use ndarray_linalg::close_l1;

use crate::parameters::Parameters;
use crate::training::Training;

use crate::training::rotation::{Rotator};









#[test]
fn test_rotation_matrix() {
    let rotation_matrix: Arc<Mutex<Option<Array3<f32>>>> = Arc::new(Mutex::new(None));
    let rotation_matrix_clone = rotation_matrix.clone();

    let expects = arr3(&[
         [[ 1.39886206e-03,  3.49516576e-03],
          [ 9.43944169e-04,  8.26974144e-03],
          [ 9.99998576e-01,  9.99959697e-01]],
         [[ 9.43944169e-04,  8.26974144e-03],
          [ 9.99999108e-01,  9.99931372e-01],
          [-9.45265121e-04, -8.29841247e-03]],
         [[-9.99998576e-01, -9.99959697e-01],
          [ 9.45265121e-04,  8.29841247e-03],
          [ 1.39796979e-03,  3.42653727e-03]]]);

    let _system = System::run(move || {
        let mut training = Training::new(Parameters::default());
        let dummy_data = arr3(&[[[0.]]]);

        training.rotation.phase_space = Some(dummy_data.to_shared());
        training.rotation.data_ref = Some(dummy_data.to_shared());

        training.rotation.reduced_ref = Some(arr3(&[
            [[-2.32510113e+01, -1.84500066e+01],
             [ 2.19784013e-02,  1.53111935e-01],
             [ 3.25042576e-02,  6.32221831e-02]]
        ]));

        *(rotation_matrix_clone.lock().unwrap()) = Some(training.get_rotation_matrix());
        System::current().stop();
    });
    let truth = rotation_matrix.lock().unwrap();

    close_l1(truth.as_ref().unwrap(), &expects, 0.0005)
}

/*struct TestParams {
    ip: SocketAddr,
    seeds: Vec<SocketAddr>,
    other_nodes: Vec<(usize, SocketAddr)>,
    main: bool,
    data: ArcArray2<f32>,
    expected: Array2<f32>
}

#[test]
#[ignore]
fn test_distributed_rotation() {
    let ip1: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000)).parse().unwrap();
    let ip2: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000)).parse().unwrap();


    let dataset = read_data_("data/test.csv");
    let expected: Array2<f32> = arr2(&[
        [0.7265024, -0.39373094, 0.5631784],
        [0.57647973, -0.09682596, -0.8113543]
    ]);

    let arr = [
        TestParams {
            ip: ip1.clone(),
            seeds: vec![],
            other_nodes: vec![(1, ip2.clone())],
            main: true,
            data: dataset.slice(s![..50, ..]).to_shared(),
            expected: expected.clone()
        },
        TestParams {
            ip: ip2.clone(),
            seeds: vec![ip1.clone()],
            other_nodes: vec![(0, ip1.clone())],
            main: false,
            data: dataset.slice(s![50.., ..]).to_shared(),
            expected: expected
        },
    ];
    arr.into_par_iter().for_each(|p| run_single_node_rotation(p.ip, p.seeds.clone(), p.other_nodes, p.main, p.data, p.expected));
}


#[actix_rt::main]
async fn run_single_node_rotation(ip_address: SocketAddr, seed_nodes: Vec<SocketAddr>, other_nodes: Vec<(usize, SocketAddr)>, main: bool, data: ArcArray2<f32>, expected: Array2<f32>) {

    let arc_cluster_nodes = Arc::new(Mutex::new(None));
    let cloned_arc_cluster_nodes = arc_cluster_nodes.clone();
    //let result = Arc::new(Mutex::new(None));
    //let cloned = Arc::clone(&result);

    delay_for(Duration::from_millis(200)).await;

    let _cluster = Cluster::new(ip_address.clone(), seed_nodes.clone());
    if seed_nodes.len() > 0 {
        let _cluster_listener = TestClusterMemberListener::new(main, seed_nodes[0], other_nodes.len() + 1, ip_address, cloned_arc_cluster_nodes).start();
    } else {
        let _cluster_listener = TestClusterMemberListener::new(main, ip_address, other_nodes.len() + 1, ip_address, cloned_arc_cluster_nodes).start();
    }

    delay_for(Duration::from_millis(200)).await;

    let cluster_nodes: ClusterNodes = (*arc_cluster_nodes.lock().unwrap()).as_ref().unwrap().clone();

    let dummy_data = arr3(&[[[0.]]]);

    let parameters = Parameters::default();
    let mut training = Training::new(parameters);
    training.cluster_nodes = cluster_nodes;
    let training_addr = training.start();
    training_addr.do_send(DataLoadedAndProcessed {
        data_ref: dummy_data.to_shared(),
        phase_space: dummy_data.to_shared(),
        dataset_stats: Default::default()
    });

    delay_for(Duration::from_millis(3000)).await;

    println!("done");
}*/
