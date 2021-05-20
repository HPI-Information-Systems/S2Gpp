use crate::data_manager::data_reader::read_data_;
use crate::parameters::{Parameters, Role};
use ndarray::{arr2, s};
use ndarray_linalg::close_l1;
use crate::data_manager::reference_dataset_builder::ReferenceDatasetBuilder;

#[test]
fn test_correct_spacing() {
    let data = read_data_("data/test.csv");
    let parameters = Parameters {
        role: Role::Main { data_path: "data/test.csv".to_string() },
        local_host: "127.0.0.1:8000".parse().unwrap(),
        pattern_length: 50,
        latent: 16,
        n_threads: 20,
        n_cluster_nodes: 1
    };

    let rdb = ReferenceDatasetBuilder::new(data.to_shared(), parameters);
    let df_ref = rdb.build();
    let expected = arr2(&[
          [ 0.09360752,  0.63319328,  0.0458456 ],
          [ 0.09360752,  0.63319328,  0.0458456 ],
          [ 0.09360752,  0.63319328,  0.0458456 ]]);

    close_l1(&df_ref.slice(s![0, 0..3, 0..3]), &expected, 0.0005)
}
