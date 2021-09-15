use crate::training::Training;
use crate::parameters::Parameters;
use ndarray::arr1;
use crate::training::scoring::Scorer;
use std::path::Path;
use std::fs::remove_file;

#[test]
fn scores_are_written_to_file() {
    let mut training = Training::new(Parameters::default());
    training.scoring.score = Some(arr1(&[1.,1.,1.,1.,1.]));

    let scores_path = "data/_test_scores.csv";
    training.output_score(scores_path.to_string()).unwrap();
    let path = Path::new(scores_path);
    assert!(path.exists());
    remove_file(path).expect("Could not delete test file!");
}
