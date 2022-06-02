use crate::data_manager::data_reader::read_data_;
use crate::{Parameters, SyncInterface, Training};

#[test]
fn test_interface_for_training_actor() {
    let parameters = Parameters::default();
    let mut s2gpp = Training::init(parameters);

    let dataset = read_data_("data/ts_0.csv");
    let anomaly_score = s2gpp.fit(dataset);
    assert!(anomaly_score.is_ok());
}
