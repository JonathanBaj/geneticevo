use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::model::{Architecture, Model};

type GraphSpec = (usize, Vec<(usize, usize)>, usize);
type ModelSpec = (Vec<usize>, Option<GraphSpec>);

struct Member {
    model: Model,
    weights: Vec<f32>,
    input: Vec<f32>,
}

#[pyclass(module = "_geneticevo_rust")]
pub(crate) struct PolicyPopulation {
    members: Vec<Member>,
    observation_dim: usize,
    discrete: bool,
    low: Vec<f32>,
    scale: Vec<f32>,
}

fn invalid(message: &'static str) -> PyErr {
    PyValueError::new_err(message)
}

fn ordered<'a, T: numpy::Element, D: numpy::ndarray::Dimension>(
    array: &'a numpy::PyReadonlyArray<'_, T, D>,
) -> PyResult<&'a [T]> {
    if !array.is_c_contiguous() {
        return Err(invalid("Expected a C-contiguous NumPy array"));
    }
    array
        .as_slice()
        .map_err(|_| invalid("Expected an aligned NumPy array"))
}

fn evaluate<T: Copy + Default + Send>(
    members: &mut [Member],
    size: usize,
    operation: impl Fn(&mut Member, &mut [T]) + Send + Sync,
) -> Vec<T> {
    let mut output = vec![T::default(); members.len() * size];
    if members.len() < 4 {
        members
            .iter_mut()
            .zip(output.chunks_exact_mut(size))
            .for_each(|(member, target)| operation(member, target));
    } else {
        members
            .par_iter_mut()
            .zip(output.par_chunks_mut(size))
            .with_min_len(4)
            .for_each(|(member, target)| operation(member, target));
    }
    output
}

impl PolicyPopulation {
    fn forward_inputs(&mut self, obs: &PyReadonlyArray3<'_, f32>) -> PyResult<usize> {
        let [population, rows, dimensions] = obs.shape() else {
            unreachable!()
        };
        if *population != self.members.len()
            || *population == 0
            || *rows == 0
            || *dimensions != self.observation_dim
            || rows
                .checked_mul(self.low.len())
                .and_then(|n| n.checked_mul(*population))
                .is_none()
        {
            return Err(invalid(
                "Observations must have shape (members, copies, observation_dim)",
            ));
        }
        // Take an owned snapshot before releasing the GIL.
        let inputs = ordered(obs)?;
        for (member, input) in self
            .members
            .iter_mut()
            .zip(inputs.chunks_exact(rows * self.observation_dim))
        {
            member.input.clear();
            member.input.extend_from_slice(input);
        }
        Ok(*rows)
    }
}

#[pymethods]
impl PolicyPopulation {
    #[new]
    fn new(
        observation_dim: usize,
        discrete: bool,
        low: Vec<f32>,
        high: Vec<f32>,
    ) -> PyResult<Self> {
        let scale: Vec<_> = low
            .iter()
            .zip(&high)
            .map(|(&a, &b)| ((f64::from(b) - f64::from(a)) * 0.5) as f32)
            .collect();
        if observation_dim == 0
            || low.is_empty()
            || low.len() != high.len()
            || (!discrete
                && (!low.iter().chain(&high).all(|x| x.is_finite())
                    || scale.iter().any(|&s| s <= 0.0 || !s.is_finite())))
        {
            return Err(invalid("Invalid observation or action space"));
        }
        Ok(Self {
            members: Vec::new(),
            observation_dim,
            discrete,
            low,
            scale,
        })
    }

    fn add(&mut self, spec: ModelSpec, genes: PyReadonlyArray1<'_, f32>) -> PyResult<()> {
        let weights = ordered(&genes)?;
        let (hidden, graph) = spec;
        let architecture = if let Some((nodes, edges, settling_steps)) = graph {
            Architecture::Graph {
                nodes,
                edges,
                settling_steps,
            }
        } else {
            Architecture::Dense(hidden)
        };
        let model = Model::new(
            self.observation_dim,
            self.low.len(),
            self.discrete,
            architecture,
        )
        .map_err(invalid)?;
        if model.parameter_count != weights.len() || !weights.iter().all(|x| x.is_finite()) {
            return Err(invalid("Incompatible or non-finite genes"));
        }
        self.members.push(Member {
            model,
            weights: weights.to_vec(),
            input: Vec::new(),
        });
        Ok(())
    }

    fn act<'py>(
        &mut self,
        py: Python<'py>,
        obs: PyReadonlyArray3<'_, f32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rows = self.forward_inputs(&obs)?;
        let members = self.members.len();
        let (low, scale) = (&self.low, &self.scale);
        let actions = low.len();
        if self.discrete {
            let result = py.detach(|| {
                evaluate(&mut self.members, rows, |member, target| {
                    let values = member.model.actor(&member.weights, &member.input, rows);
                    for (out, row) in target
                        .iter_mut()
                        .zip(values.chunks_exact(values.len() / rows))
                    {
                        let mut best = 0;
                        for (i, &value) in row[..actions].iter().enumerate().skip(1) {
                            if value > row[best] {
                                best = i;
                            }
                        }
                        *out = best as i64;
                    }
                })
            });
            Ok(Array2::from_shape_vec((members, rows), result)
                .unwrap()
                .into_pyarray(py)
                .into_any())
        } else {
            let result = py.detach(|| {
                evaluate(&mut self.members, rows * actions, |member, target| {
                    let values = member.model.actor(&member.weights, &member.input, rows);
                    for (out, row) in target
                        .chunks_exact_mut(actions)
                        .zip(values.chunks_exact(values.len() / rows))
                    {
                        for (i, output) in out.iter_mut().enumerate() {
                            *output = low[i] + (row[i].tanh() + 1.0) * scale[i];
                        }
                    }
                })
            });
            Ok(Array3::from_shape_vec((members, rows, actions), result)
                .unwrap()
                .into_pyarray(py)
                .into_any())
        }
    }

    #[pyo3(signature = (obs, critic=false))]
    fn forward<'py>(
        &mut self,
        py: Python<'py>,
        obs: PyReadonlyArray3<'_, f32>,
        critic: bool,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let rows = self.forward_inputs(&obs)?;
        let actions = self.low.len();
        let columns = if critic { 1 } else { actions };
        let result = py.detach(|| {
            evaluate(&mut self.members, rows * columns, |member, target| {
                let offset = if critic && !member.model.separate_critic() {
                    actions
                } else {
                    0
                };
                let values = if critic {
                    member.model.critic(&member.weights, &member.input, rows)
                } else {
                    member.model.actor(&member.weights, &member.input, rows)
                };
                for (out, row) in target
                    .chunks_exact_mut(columns)
                    .zip(values.chunks_exact(values.len() / rows))
                {
                    out.copy_from_slice(&row[offset..offset + columns]);
                }
            })
        });
        Ok(
            Array3::from_shape_vec((self.members.len(), rows, columns), result)
                .unwrap()
                .into_pyarray(py),
        )
    }
}
