use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods, ToPyArray,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::model::{Architecture, Model};
use crate::ppo::{Adam, Member, RngState, Settings, Space, METRICS};

type GraphSpec = (usize, Vec<(usize, usize)>, usize);
type ModelSpec = (Vec<usize>, Option<GraphSpec>);
type AdamInput<'py> = (u64, PyReadonlyArray1<'py, f32>, PyReadonlyArray1<'py, f32>);
type MemberState<'py> = (
    Bound<'py, PyArray1<f32>>,
    u64,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    RngState,
);

#[pyclass(module = "_geneticevo_rust")]
pub(crate) struct Population {
    members: Vec<Member>,
    space: Space,
    settings: Settings,
    step: usize,
    pending: bool,
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
        .map_err(|_| invalid("Expected a contiguous and aligned NumPy array"))
}

fn for_each_member(members: &mut [Member], operation: impl Fn(&mut Member) + Send + Sync) {
    if members.len() < 4 {
        members.iter_mut().for_each(operation);
    } else {
        members.par_iter_mut().with_min_len(4).for_each(operation);
    }
}

impl Population {
    fn at_boundary(&self) -> PyResult<()> {
        if self.step != 0 || self.pending {
            return Err(invalid(
                "Member edits and exports require a completed rollout",
            ));
        }
        Ok(())
    }

    fn model(&self, spec: ModelSpec, weights: &[f32]) -> PyResult<Model> {
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
            self.space.observations,
            self.space.actions,
            self.space.discrete,
            architecture,
        )
        .map_err(invalid)?;
        if model.parameter_count != weights.len() || !weights.iter().all(|x| x.is_finite()) {
            return Err(invalid("Incompatible or non-finite genes"));
        }
        Ok(model)
    }

    fn observations(&self, obs: &PyReadonlyArray3<'_, f32>) -> PyResult<usize> {
        let [population, rows, dimensions] = obs.shape() else {
            unreachable!()
        };
        if *population != self.members.len()
            || *population == 0
            || *rows == 0
            || *dimensions != self.space.observations
            || rows
                .checked_mul(self.space.actions)
                .and_then(|n| n.checked_mul(*population))
                .is_none()
        {
            return Err(invalid(
                "Observations must have shape (members, copies, observation_dim)",
            ));
        }
        Ok(*rows)
    }

    fn forward_inputs(&mut self, obs: &PyReadonlyArray3<'_, f32>) -> PyResult<usize> {
        let rows = self.observations(obs)?;
        let inputs = ordered(obs)?;
        for (member, input) in self
            .members
            .iter_mut()
            .zip(inputs.chunks_exact(rows * self.space.observations))
        {
            member.step_input.clear();
            member.step_input.extend_from_slice(input);
        }
        Ok(rows)
    }

    fn actions<'py>(&self, py: Python<'py>, rows: usize) -> Bound<'py, PyAny> {
        let shape = (self.members.len(), rows);
        if self.space.discrete {
            let actions = self
                .members
                .iter()
                .flat_map(|m| m.discrete_output.iter().copied())
                .collect();
            Array2::from_shape_vec(shape, actions)
                .unwrap()
                .into_pyarray(py)
                .into_any()
        } else {
            let actions = self
                .members
                .iter()
                .flat_map(|m| m.continuous_output.iter().copied())
                .collect();
            Array3::from_shape_vec((shape.0, shape.1, self.space.actions), actions)
                .unwrap()
                .into_pyarray(py)
                .into_any()
        }
    }
}

#[pymethods]
impl Population {
    #[new]
    fn new(
        observation_dim: usize,
        discrete: bool,
        low: Vec<f32>,
        high: Vec<f32>,
        settings: Settings,
    ) -> PyResult<Self> {
        let actions = low.len();
        let scale: Vec<_> = low
            .iter()
            .zip(&high)
            .map(|(&a, &b)| {
                // Compute the range without overflowing for finite float32 bounds.
                ((f64::from(b) - f64::from(a)) * 0.5) as f32
            })
            .collect();
        let count = settings.count().unwrap_or(0);
        if observation_dim == 0
            || actions == 0
            || actions != high.len()
            || !settings.validate()
            || count.checked_mul(observation_dim).is_none()
            || count.checked_mul(actions).is_none()
            || settings
                .epochs
                .checked_mul(count.div_ceil(settings.minibatch_size))
                .is_none()
            || (!discrete
                && (!low.iter().chain(&high).all(|x| x.is_finite())
                    || scale.iter().any(|&s| s <= 0.0 || !s.is_finite())))
        {
            return Err(invalid("Invalid policy spaces or PPO settings"));
        }
        let log_scale = scale.iter().map(|s| s.ln()).collect();
        let space = Space {
            observations: observation_dim,
            actions,
            discrete,
            low,
            scale,
            log_scale,
        };
        Ok(Self {
            members: Vec::new(),
            space,
            settings,
            step: 0,
            pending: false,
        })
    }

    #[pyo3(signature = (spec, genes, rng, adam=None))]
    fn add(
        &mut self,
        spec: ModelSpec,
        genes: PyReadonlyArray1<'_, f32>,
        rng: RngState,
        adam: Option<AdamInput<'_>>,
    ) -> PyResult<()> {
        self.at_boundary()?;
        let weights = ordered(&genes)?;
        let model = self.model(spec, weights)?;
        let optimizer = if let Some((step, first, second)) = adam {
            let (first, second) = (ordered(&first)?, ordered(&second)?);
            if first.len() != second.len()
                || (first.len() != weights.len() && !(step == 0 && first.is_empty()))
                || !first.iter().all(|x| x.is_finite())
                || !second.iter().all(|x| x.is_finite() && *x >= 0.0)
            {
                return Err(invalid("Invalid Adam checkpoint state"));
            }
            Adam {
                step,
                first: first.to_vec(),
                second: second.to_vec(),
            }
        } else {
            Adam::default()
        };
        self.members
            .push(Member::new(model, weights.to_vec(), optimizer, rng));
        Ok(())
    }

    fn get_genes<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.at_boundary()?;
        let member = self
            .members
            .get(index)
            .ok_or_else(|| invalid("Invalid member index"))?;
        Ok(member.weights.to_pyarray(py))
    }

    fn replace_topology(
        &mut self,
        index: usize,
        spec: ModelSpec,
        genes: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<()> {
        self.at_boundary()?;
        let weights = ordered(&genes)?;
        let model = self.model(spec, weights)?;
        let member = self
            .members
            .get_mut(index)
            .ok_or_else(|| invalid("Invalid member index"))?;
        member.model = model;
        member.weights.clear();
        member.weights.extend_from_slice(weights);
        member.adam = Adam::default();
        Ok(())
    }

    fn export<'py>(&self, py: Python<'py>) -> PyResult<Vec<MemberState<'py>>> {
        self.at_boundary()?;
        Ok(self
            .members
            .iter()
            .map(|member| {
                (
                    member.weights.to_pyarray(py),
                    member.adam.step,
                    member.adam.first.to_pyarray(py),
                    member.adam.second.to_pyarray(py),
                    member.rng_state(),
                )
            })
            .collect())
    }

    fn act<'py>(
        &mut self,
        py: Python<'py>,
        obs: PyReadonlyArray3<'_, f32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rows = self.forward_inputs(&obs)?;
        let space = &self.space;
        py.detach(|| {
            for_each_member(&mut self.members, |member| {
                member.actions(space, rows, None)
            })
        });
        Ok(self.actions(py, rows))
    }

    #[pyo3(signature = (obs, critic=false))]
    fn forward<'py>(
        &mut self,
        py: Python<'py>,
        obs: PyReadonlyArray3<'_, f32>,
        critic: bool,
    ) -> PyResult<Bound<'py, numpy::PyArray3<f32>>> {
        let rows = self.forward_inputs(&obs)?;
        let actions = self.space.actions;
        let columns = if critic { 1 } else { actions };
        let mut output = vec![0.0; self.members.len() * rows * columns];
        py.detach(|| {
            self.members
                .par_iter_mut()
                .zip(output.par_chunks_mut(rows * columns))
                .with_min_len(4)
                .for_each(|(member, target)| {
                    let separate = member.model.separate_critic();
                    let values = if critic {
                        member
                            .model
                            .critic(&member.weights, &member.step_input, rows)
                    } else {
                        member
                            .model
                            .actor(&member.weights, &member.step_input, rows)
                    };
                    let offset = if critic && !separate { actions } else { 0 };
                    for (target, values) in target
                        .chunks_exact_mut(columns)
                        .zip(values.chunks_exact(values.len() / rows))
                    {
                        target.copy_from_slice(&values[offset..offset + columns]);
                    }
                });
        });
        Ok(
            Array3::from_shape_vec((self.members.len(), rows, columns), output)
                .unwrap()
                .into_pyarray(py),
        )
    }

    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        obs: PyReadonlyArray3<'_, f32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if self.pending || self.step == self.settings.rollout_steps {
            return Err(invalid(
                "Record the pending transition or train on the complete rollout first",
            ));
        }
        let rows = self.observations(&obs)?;
        if rows != self.settings.env_copies {
            return Err(invalid(
                "Observations must match the configured environment copies",
            ));
        }
        let source = ordered(&obs)?;
        let offset = self.step * rows * self.space.observations;
        let span = offset..offset + rows * self.space.observations;
        for (member, input) in self
            .members
            .iter_mut()
            .zip(source.chunks_exact(rows * self.space.observations))
        {
            if self.step == 0 {
                member
                    .rollout
                    .resize(self.settings.count().unwrap(), &self.space);
            }
            member.rollout.obs[span.clone()].copy_from_slice(input);
        }
        let (space, start) = (&self.space, self.step * rows);
        py.detach(|| {
            for_each_member(&mut self.members, |member| {
                member.actions(space, rows, Some(start))
            })
        });
        self.pending = true;
        Ok(self.actions(py, rows))
    }

    fn record(
        &mut self,
        rewards: PyReadonlyArray2<'_, f32>,
        terminated: PyReadonlyArray2<'_, bool>,
        truncated: PyReadonlyArray2<'_, bool>,
        next: PyReadonlyArray3<'_, f32>,
    ) -> PyResult<()> {
        let rows = self.observations(&next)?;
        let shape = [self.members.len(), self.settings.env_copies];
        if !self.pending
            || rows != shape[1]
            || rewards.shape() != shape
            || terminated.shape() != shape
            || truncated.shape() != shape
        {
            return Err(invalid(
                "Record one matching transition after each sample call",
            ));
        }
        let (rewards, terminated, truncated, next) = (
            ordered(&rewards)?,
            ordered(&terminated)?,
            ordered(&truncated)?,
            ordered(&next)?,
        );
        let dim = self.space.observations;
        let start = self.step * rows;
        let last = self.step + 1 == self.settings.rollout_steps;
        for (i, member) in self.members.iter_mut().enumerate() {
            let slot = start..start + rows;
            let source = i * rows..(i + 1) * rows;
            member.rollout.rewards[slot.clone()].copy_from_slice(&rewards[source.clone()]);
            member.rollout.terminated[slot.clone()].copy_from_slice(&terminated[source.clone()]);
            member.rollout.truncated[slot].copy_from_slice(&truncated[source.clone()]);
            for (row, item) in source.enumerate() {
                if !terminated[item] && (last || truncated[item]) {
                    member.rollout.final_obs[(start + row) * dim..(start + row + 1) * dim]
                        .copy_from_slice(&next[item * dim..(item + 1) * dim]);
                }
            }
        }
        self.step += 1;
        self.pending = false;
        Ok(())
    }

    fn update<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if self.pending || self.step != self.settings.rollout_steps {
            return Err(invalid(
                "Cannot update until the rollout has been completely recorded",
            ));
        }
        let (space, settings) = (&self.space, &self.settings);
        let train = |member: &mut Member| member.train(space, settings);
        let metrics: Vec<_> = py.detach(|| {
            if self.members.len() < 4 {
                self.members.iter_mut().map(train).collect()
            } else {
                self.members
                    .par_iter_mut()
                    .with_min_len(2)
                    .map(train)
                    .collect()
            }
        });
        self.step = 0;
        Ok(Array2::from_shape_vec(
            (self.members.len(), METRICS),
            metrics.into_iter().flatten().collect(),
        )
        .unwrap()
        .into_pyarray(py))
    }
}
