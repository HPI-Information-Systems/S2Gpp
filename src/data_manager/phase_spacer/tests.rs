use crate::data_manager::data_reader::read_data_;
use crate::data_manager::phase_spacer::PhaseSpacer;
use crate::parameters::{Parameters, Role};
use ndarray::{arr2, s};
use ndarray_linalg::close_l1;

#[test]
fn test_correct_spacing() {
    let data = read_data_("data/test.csv");
    let parameters = Parameters {
        role: Role::Main {
            data_path: Some("data/test.csv".to_string()),
        },
        local_host: "127.0.0.1:8000".parse().unwrap(),
        pattern_length: 50,
        latent: 16,
        rate: 100,
        n_threads: 20,
        n_cluster_nodes: 1,
        ..Default::default()
    };

    let ps = PhaseSpacer::new(data.to_shared(), parameters);
    let phase_space = ps.build();
    let expected = arr2(&[
        [6.54935769, 9.29735908, 7.89886824],
        [7.14322411, 8.69207521, 8.13002231],
        [6.85200249, 8.12101571, 7.80578016],
    ]);

    close_l1(
        &phase_space.slice(s![0_usize, 0..3, 0..3]),
        &expected,
        0.0005,
    )
}
