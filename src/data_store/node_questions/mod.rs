use serde::{Serialize, Deserialize};
use serde_with::serde_as;

use std::collections::HashMap;
use std::iter::FromIterator;
use num_integer::Integer;
use crate::data_store::node_questions::node_in_question::NodeInQuestion;
use crate::data_store::transition::{Transition, TransitionMixin};
use crate::Parameters;

pub(crate) mod node_in_question;


#[serde_as]
#[derive(Default, Clone, Serialize, Deserialize)]
pub(crate) struct NodeQuestions {
    #[serde_as(as = "Vec<(_, Vec<(_, _)>)>")]
    node_questions: HashMap<usize, HashMap<usize, Vec<NodeInQuestion>>>
}


impl NodeQuestions {
    pub fn add_niq(&mut self, answering_node: usize, asking_node: usize, niq: NodeInQuestion) {
        if asking_node != answering_node {
            match self.node_questions.get_mut(&answering_node) {
                Some(questions) => match questions.get_mut(&asking_node) {
                    Some(nodes) => {
                        nodes.push(niq);
                        nodes.dedup();
                    },
                    None => { questions.insert(asking_node, vec![niq]); }
                },
                None => {
                    let questions = HashMap::from_iter([(asking_node, vec![niq])]);
                    self.node_questions.insert(answering_node.clone(), questions);
                }
            }
        }
    }

    pub fn remove(&mut self, answering_node: &usize) -> Option<HashMap<usize, Vec<NodeInQuestion>>> {
        self.node_questions.remove(answering_node)
    }

    pub fn clear(&mut self) {
        self.node_questions.clear()
    }

    pub fn ask(&mut self, transition: &Transition, prev_transition: Option<Transition>, within_transition: bool, cluster_span: usize, parameters: Parameters) {
        let point_id = transition.get_from_id();
        let nodes_in_question = if let Some(prev) = prev_transition {
            if within_transition {
                (0..cluster_span).into_iter().map(|i| {
                    let wanted_segment = parameters.first_segment_of_i_next_cluster_node(transition.get_from_segment(), i+1);
                    let segment_before_wanted = (wanted_segment as isize - 1).mod_floor(&(parameters.rate as isize)) as usize;

                    // if transition starts in last assigned segment
                    let (prev_point_id, prev_segment_id) = if (wanted_segment == transition.get_first_intersection_segment(&parameters.rate)) & (i == 0) {
                        // foreign should receive node from transition before
                        (prev.get_from_id(), prev.get_to_segment())
                    } else {
                        // foreign should receive node from this transition but from intersection before
                        (point_id, segment_before_wanted)
                    };

                    NodeInQuestion::new(
                        prev_point_id,
                        prev_segment_id,
                        point_id,
                        wanted_segment.mod_floor(&parameters.rate)
                    )
                }).collect()
            } else {
                vec![NodeInQuestion::new(
                    prev.get_from_id(),
                    prev.get_to_segment(),
                    point_id,
                    transition.get_first_intersection_segment(&parameters.rate)
                )]
            }
        } else {
            if within_transition {
                (0..cluster_span).into_iter().map(|i| {
                    let wanted_segment = parameters.first_segment_of_i_next_cluster_node(transition.get_from_segment(), i+1);
                    let segment_before_wanted = (wanted_segment as isize - 1).mod_floor(&(parameters.rate as isize)) as usize;

                    NodeInQuestion::new(
                        point_id,
                        segment_before_wanted,
                        point_id,
                        wanted_segment.mod_floor(&parameters.rate)
                    )
                }).collect()
            } else {
                panic!("It does happen, please solve!")
            }
        };

        for node_in_question in nodes_in_question {
            let asking_node = parameters.segment_id_to_assignment(node_in_question.get_prev_segment());
            let answering_node = parameters.segment_id_to_assignment(node_in_question.get_segment());
            self.add_niq(answering_node, asking_node, node_in_question);
        }
    }
}
