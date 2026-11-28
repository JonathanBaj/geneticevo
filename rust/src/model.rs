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
}

pub(crate) struct Model {
    actor: Network,
    critic: Option<Network>,
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
}

impl Network {
    fn new(layout: Layout, stages: usize) -> Self {
        Self {
            layout,
            activations: vec![Vec::new(); stages],
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
                        source,
                        &weights[layer.weights..layer.bias],
                        target,
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
}

/// Multiply contiguous row-major matrices.
fn multiply([rows, inner, columns]: [usize; 3], a: &[f32], b: &[f32], out: &mut [f32]) {
    if rows.saturating_mul(inner).saturating_mul(columns) >= 4096 {
        assert_eq!(a.len(), rows * inner);
        assert_eq!(b.len(), inner * columns);
        assert_eq!(out.len(), rows * columns);
        // The checked slices match the matrix sizes, and the output is uniquely borrowed.
        unsafe {
            sgemm(
                rows,
                inner,
                columns,
                1.0,
                a.as_ptr(),
                inner as isize,
                1,
                b.as_ptr(),
                columns as isize,
                1,
                0.0,
                out.as_mut_ptr(),
                columns as isize,
                1,
            );
        }
    } else {
        out.fill(0.0);
        for (input, row) in a.chunks_exact(inner).zip(out.chunks_exact_mut(columns)) {
            for (&factor, rhs) in input.iter().zip(b.chunks_exact(columns)) {
                for (output, &value) in row.iter_mut().zip(rhs) {
                    *output += factor * value;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_policy_reads_heads_at_their_genome_offsets() {
        let mut model = Model::new(2, 2, false, Architecture::Dense(vec![])).unwrap();
        let genes = [1.0, 2.0, -1.0, 0.5, 0.5, -1.0, 0.25, -2.0, 0.5, 0.0, 0.0];
        assert_eq!(model.parameter_count, genes.len());
        assert_eq!(model.actor(&genes, &[2.0, -1.0], 1), &[3.5, 2.5]);
        assert_eq!(model.critic(&genes, &[2.0, -1.0], 1), &[3.0]);
    }
}
