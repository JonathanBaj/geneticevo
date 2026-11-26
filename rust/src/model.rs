use matrixmultiply::sgemm;

struct Layer {
    input: usize,
    output: usize,
    weights: usize,
    bias: usize,
}

pub(crate) enum Architecture {
    Dense(Vec<usize>),
    Graph {
        nodes: usize,
        edges: Vec<(usize, usize)>,
        settling_steps: usize,
    },
}

enum Layout {
    Dense(Vec<Layer>),
    Graph {
        inputs: usize,
        nodes: usize,
        outputs: usize,
        edges: Vec<(usize, usize)>,
    },
}

struct Network {
    layout: Layout,
    activations: Vec<Vec<f32>>,
    deltas: [Vec<f32>; 2],
}

pub(crate) struct Model {
    actor: Network,
    critic: Option<Network>,
    joined_delta: Vec<f32>,
    pub(crate) parameter_count: usize,
}

impl Model {
    pub(crate) fn new(
        inputs: usize,
        actions: usize,
        discrete: bool,
        architecture: Architecture,
    ) -> Result<Self, &'static str> {
        let (actor, critic, weights) = match architecture {
            Architecture::Dense(hidden) => {
                let (actor, offset) = Network::dense(inputs, &hidden, actions, 0)?;
                let (critic, weights) = Network::dense(inputs, &hidden, 1, offset)?;
                (actor, Some(critic), weights)
            }
            Architecture::Graph {
                nodes,
                edges,
                settling_steps,
            } => {
                let outputs = actions.checked_add(1).ok_or("Too many actions")?;
                let sources = inputs.checked_add(nodes).ok_or("Too many graph nodes")?;
                if nodes < outputs
                    || settling_steps == 0
                    || edges
                        .iter()
                        .any(|&(from, to)| from >= sources || to >= nodes)
                {
                    return Err("Invalid graph topology");
                }
                let weights = edges
                    .len()
                    .checked_add(nodes)
                    .ok_or("Too many graph parameters")?;
                let layout = Layout::Graph {
                    inputs,
                    nodes,
                    outputs,
                    edges,
                };
                (Network::new(layout, settling_steps), None, weights)
            }
        };
        let parameter_count = weights
            .checked_add(if discrete { 0 } else { actions })
            .ok_or("Too many parameters")?;
        Ok(Self {
            actor,
            critic,
            joined_delta: Vec::new(),
            parameter_count,
        })
    }

    /// The graph shares all readouts. Its values are the last column.
    pub(crate) fn actor(&mut self, weights: &[f32], obs: &[f32], rows: usize) -> &[f32] {
        self.actor.forward(weights, obs, rows)
    }

    pub(crate) fn critic(&mut self, weights: &[f32], obs: &[f32], rows: usize) -> &[f32] {
        self.critic
            .as_mut()
            .unwrap_or(&mut self.actor)
            .forward(weights, obs, rows)
    }

    pub(crate) fn separate_critic(&self) -> bool {
        self.critic.is_some()
    }

    pub(crate) fn backward(
        &mut self,
        weights: &[f32],
        obs: &[f32],
        rows: usize,
        actor_signal: &[f32],
        critic_signal: &[f32],
        gradient: &mut [f32],
    ) {
        if let Some(critic) = &mut self.critic {
            self.actor
                .backward(weights, obs, rows, actor_signal, gradient);
            critic.backward(weights, obs, rows, critic_signal, gradient);
        } else {
            let actions = actor_signal.len() / rows;
            self.joined_delta.clear();
            for (logits, &value) in actor_signal.chunks_exact(actions).zip(critic_signal) {
                self.joined_delta.extend_from_slice(logits);
                self.joined_delta.push(value);
            }
            self.actor
                .backward(weights, obs, rows, &self.joined_delta, gradient);
        }
    }
}

impl Network {
    fn new(layout: Layout, stages: usize) -> Self {
        Self {
            layout,
            activations: vec![Vec::new(); stages],
            deltas: [Vec::new(), Vec::new()],
        }
    }

    fn dense(
        inputs: usize,
        hidden: &[usize],
        outputs: usize,
        mut offset: usize,
    ) -> Result<(Self, usize), &'static str> {
        let mut layers = Vec::with_capacity(hidden.len() + 1);
        let mut input = inputs;
        for output in hidden.iter().copied().chain([outputs]) {
            if output == 0 {
                return Err("Layer widths must be positive");
            }
            let bias = input
                .checked_mul(output)
                .and_then(|count| offset.checked_add(count))
                .ok_or("Too many dense parameters")?;
            layers.push(Layer {
                input,
                output,
                weights: offset,
                bias,
            });
            offset = bias
                .checked_add(output)
                .ok_or("Too many dense parameters")?;
            input = output;
        }
        let stages = layers.len();
        Ok((Self::new(Layout::Dense(layers), stages), offset))
    }

    fn forward(&mut self, weights: &[f32], obs: &[f32], rows: usize) -> &[f32] {
        match &self.layout {
            Layout::Dense(layers) => {
                for (i, layer) in layers.iter().enumerate() {
                    let (earlier, current) = self.activations.split_at_mut(i);
                    let source = earlier.last().map_or(obs, Vec::as_slice);
                    let target = &mut current[0];
                    target.resize(rows * layer.output, 0.0);
                    multiply(
                        [rows, layer.input, layer.output],
                        [false, false],
                        source,
                        &weights[layer.weights..layer.bias],
                        target,
                        false,
                    );
                    let biases = &weights[layer.bias..layer.bias + layer.output];
                    for row in target.chunks_exact_mut(layer.output) {
                        for (unit, bias) in row.iter_mut().zip(biases) {
                            *unit += bias;
                        }
                        if i + 1 < layers.len() {
                            row.iter_mut().for_each(|unit| *unit = unit.tanh());
                        }
                    }
                }
            }
            Layout::Graph {
                inputs,
                nodes,
                outputs,
                edges,
            } => {
                let bias = &weights[edges.len()..edges.len() + nodes];
                for time in 0..self.activations.len() {
                    let (earlier, current) = self.activations.split_at_mut(time);
                    let previous = earlier.last();
                    let target = &mut current[0];
                    target.resize(rows * nodes, 0.0);
                    for (row, (input, next)) in obs
                        .chunks_exact(*inputs)
                        .zip(target.chunks_exact_mut(*nodes))
                        .enumerate()
                    {
                        next.copy_from_slice(bias);
                        for (&(source, destination), &weight) in edges.iter().zip(weights) {
                            if source < *inputs {
                                next[destination] += weight * input[source];
                            } else if let Some(state) = previous {
                                next[destination] += weight * state[row * nodes + source - inputs];
                            }
                        }
                        next[*outputs..]
                            .iter_mut()
                            .for_each(|unit| *unit = unit.tanh());
                    }
                }
            }
        }
        self.activations.last().unwrap()
    }

    fn backward(
        &mut self,
        weights: &[f32],
        obs: &[f32],
        rows: usize,
        signal: &[f32],
        gradient: &mut [f32],
    ) {
        let [current, previous] = &mut self.deltas;
        match &self.layout {
            Layout::Dense(layers) => {
                current.clear();
                current.extend_from_slice(signal);
                for (i, layer) in layers.iter().enumerate().rev() {
                    let source = i
                        .checked_sub(1)
                        .map_or(obs, |index| &self.activations[index]);
                    multiply(
                        [layer.input, rows, layer.output],
                        [true, false],
                        source,
                        current,
                        &mut gradient[layer.weights..layer.bias],
                        false,
                    );
                    let bias = &mut gradient[layer.bias..layer.bias + layer.output];
                    for row in current.chunks_exact(layer.output) {
                        for (value, delta) in bias.iter_mut().zip(row) {
                            *value += delta;
                        }
                    }
                    if i > 0 {
                        previous.resize(rows * layer.input, 0.0);
                        multiply(
                            [rows, layer.output, layer.input],
                            [false, true],
                            current,
                            &weights[layer.weights..layer.bias],
                            previous,
                            false,
                        );
                        for (delta, value) in previous.iter_mut().zip(source) {
                            *delta *= 1.0 - value * value;
                        }
                        std::mem::swap(current, previous);
                    }
                }
            }
            Layout::Graph {
                inputs,
                nodes,
                outputs,
                edges,
            } => {
                current.resize(rows * nodes, 0.0);
                current.fill(0.0);
                previous.resize(rows * nodes, 0.0);
                for (deltas, readouts) in current
                    .chunks_exact_mut(*nodes)
                    .zip(signal.chunks_exact(*outputs))
                {
                    deltas[..*outputs].copy_from_slice(readouts);
                }
                let (edge_gradient, bias_gradient) = gradient.split_at_mut(edges.len());
                for time in (0..self.activations.len()).rev() {
                    let states = &self.activations[time];
                    let preceding = time.checked_sub(1).map(|i| &self.activations[i]);
                    previous.fill(0.0);
                    for row in 0..rows {
                        let span = row * nodes..(row + 1) * nodes;
                        let delta = &mut current[span.clone()];
                        let prior_delta = &mut previous[span.clone()];
                        for (value, update) in states[span.clone()][*outputs..]
                            .iter()
                            .zip(&mut delta[*outputs..])
                        {
                            *update *= 1.0 - value * value;
                        }
                        for (bias, &update) in bias_gradient.iter_mut().zip(delta.iter()) {
                            *bias += update;
                        }
                        for (edge, &(source, target)) in edges.iter().enumerate() {
                            if source < *inputs {
                                edge_gradient[edge] += obs[row * inputs + source] * delta[target];
                            } else if let Some(preceding) = preceding {
                                edge_gradient[edge] +=
                                    preceding[span.start + source - inputs] * delta[target];
                                prior_delta[source - inputs] += weights[edge] * delta[target];
                            }
                        }
                    }
                    std::mem::swap(current, previous);
                }
            }
        }
    }
}

/// Row-major multiply with optional transposes for the two backprop products.
fn multiply(
    [rows, inner, columns]: [usize; 3],
    [transpose_a, transpose_b]: [bool; 2],
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    accumulate: bool,
) {
    let (ars, acs) = if transpose_a { (1, rows) } else { (inner, 1) };
    let (brs, bcs) = if transpose_b {
        (1, inner)
    } else {
        (columns, 1)
    };
    if rows.saturating_mul(inner).saturating_mul(columns) >= 4096 {
        assert_eq!(a.len(), rows * inner);
        assert_eq!(b.len(), inner * columns);
        assert_eq!(out.len(), rows * columns);
        // The contiguous slices have the asserted lengths. Element strides
        // describe their logical transposes, and out is a unique mutable slice.
        unsafe {
            sgemm(
                rows,
                inner,
                columns,
                1.0,
                a.as_ptr(),
                ars as isize,
                acs as isize,
                b.as_ptr(),
                brs as isize,
                bcs as isize,
                if accumulate { 1.0 } else { 0.0 },
                out.as_mut_ptr(),
                columns as isize,
                1,
            );
        }
    } else {
        if !accumulate {
            out.fill(0.0);
        }
        for (i, row) in out.chunks_exact_mut(columns).enumerate() {
            if transpose_b {
                for (j, output) in row.iter_mut().enumerate() {
                    *output += (0..inner)
                        .map(|k| a[i * ars + k * acs] * b[j * inner + k])
                        .sum::<f32>();
                }
            } else {
                for (k, rhs) in b.chunks_exact(columns).enumerate() {
                    let factor = a[i * ars + k * acs];
                    for (output, &value) in row.iter_mut().zip(rhs) {
                        *output += factor * value;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient_check(architecture: Architecture, inputs: &[f32]) {
        let mut model = Model::new(2, 2, true, architecture).unwrap();
        let mut weights: Vec<f32> = (0..model.parameter_count)
            .map(|i| ((i * 13 % 19) as f32 - 9.0) * 0.04)
            .collect();
        let (actor, value) = ([0.6, -0.3, -0.7, 0.2], [-0.1, 0.3]);
        let mut gradient = vec![0.0; weights.len()];
        model.actor(&weights, inputs, 2);
        if model.separate_critic() {
            model.critic(&weights, inputs, 2);
        }
        model.backward(&weights, inputs, 2, &actor, &value, &mut gradient);
        let loss = |model: &mut Model, weights: &[f32]| {
            let separate = model.separate_critic();
            let output = model.actor(weights, inputs, 2);
            if separate {
                let action_loss: f32 = output.iter().zip(actor).map(|(a, b)| a * b).sum();
                action_loss
                    + model
                        .critic(weights, inputs, 2)
                        .iter()
                        .zip(value)
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
            } else {
                output
                    .chunks_exact(output.len() / 2)
                    .flat_map(|row| row[..3].iter())
                    .zip([actor[0], actor[1], value[0], actor[2], actor[3], value[1]])
                    .map(|(a, b)| a * b)
                    .sum()
            }
        };
        for (i, expected) in gradient.into_iter().enumerate() {
            let initial = weights[i];
            weights[i] = initial + 1e-3;
            let upper = loss(&mut model, &weights);
            weights[i] = initial - 1e-3;
            let lower = loss(&mut model, &weights);
            weights[i] = initial;
            assert!(
                ((upper - lower) / 2e-3 - expected).abs() < 8e-5,
                "parameter {i}"
            );
        }
    }

    #[test]
    fn dense_backward_matches_finite_differences() {
        gradient_check(Architecture::Dense(vec![3, 2]), &[0.4, 0.7, -0.3, 0.9]);
        gradient_check(Architecture::Dense(vec![]), &[0.4, 0.7, -0.3, 0.9]);
    }

    #[test]
    fn cyclic_graph_backward_matches_finite_differences() {
        gradient_check(
            Architecture::Graph {
                nodes: 4,
                edges: vec![(0, 3), (1, 1), (5, 0), (5, 3), (2, 2), (0, 2)],
                settling_steps: 4,
            },
            &[0.4, 0.7, -0.3, 0.9],
        );
    }
}
