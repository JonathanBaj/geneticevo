use std::f32::consts::{LN_2, TAU};

use pyo3::FromPyObject;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use crate::model::Model;

#[derive(Clone, FromPyObject)]
pub(crate) struct Settings {
    pub env_copies: usize,
    pub rollout_steps: usize,
    pub epochs: usize,
    pub minibatch_size: usize,
    pub learning_rate: f32,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub clip: f32,
    pub value_coefficient: f32,
    pub entropy_coefficient: f32,
    pub max_gradient_norm: f32,
}

impl Settings {
    pub(crate) fn count(&self) -> Option<usize> {
        self.env_copies.checked_mul(self.rollout_steps)
    }

    pub(crate) fn validate(&self) -> bool {
        self.env_copies > 0
            && self.count().is_some_and(|count| count >= 2)
            && self.epochs > 0
            && self.minibatch_size > 0
            && (0.0..=1.0).contains(&self.gamma)
            && (0.0..=1.0).contains(&self.gae_lambda)
            && self.learning_rate.is_finite()
            && self.learning_rate > 0.0
            && self.clip.is_finite()
            && self.clip > 0.0
            && self.max_gradient_norm.is_finite()
            && self.max_gradient_norm > 0.0
            && self.value_coefficient.is_finite()
            && self.entropy_coefficient.is_finite()
    }
}

pub(crate) struct Space {
    pub observations: usize,
    pub actions: usize,
    pub discrete: bool,
    pub low: Vec<f32>,
    pub scale: Vec<f32>,
    pub log_scale: Vec<f32>,
}

#[derive(Default)]
pub(crate) struct Adam {
    pub step: u64,
    pub first: Vec<f32>,
    pub second: Vec<f32>,
}

impl Adam {
    fn step(&mut self, weights: &mut [f32], gradient: &[f32], settings: &Settings) -> f32 {
        self.step += 1;
        self.first.resize(weights.len(), 0.0);
        self.second.resize(weights.len(), 0.0);
        let size = gradient.iter().map(|g| g * g).sum::<f32>().sqrt();
        let scale = (settings.max_gradient_norm / (size + 1e-6)).min(1.0);
        let first_correction = 1.0 - 0.9_f64.powf(self.step as f64);
        let second_correction = (1.0 - 0.999_f64.powf(self.step as f64)).sqrt() as f32;
        let rate = settings.learning_rate / first_correction as f32;
        for (((weight, first), second), gradient) in weights
            .iter_mut()
            .zip(&mut self.first)
            .zip(&mut self.second)
            .zip(gradient)
        {
            let g = gradient * scale;
            *first = 0.9 * *first + 0.1 * g;
            *second = 0.999 * *second + 0.001 * g * g;
            *weight -= rate * *first / (second.sqrt() / second_correction + 1e-5);
        }
        size
    }
}

#[derive(Default)]
pub(crate) struct Rollout {
    pub obs: Vec<f32>,
    pub final_obs: Vec<f32>,
    pub rewards: Vec<f32>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
    log_probs: Vec<f32>,
    discrete: Vec<i64>,
    raw: Vec<f32>,
}

impl Rollout {
    pub(crate) fn resize(&mut self, count: usize, space: &Space) {
        self.obs.resize(count * space.observations, 0.0);
        self.final_obs.resize(count * space.observations, 0.0);
        self.rewards.resize(count, 0.0);
        self.terminated.resize(count, false);
        self.truncated.resize(count, false);
        self.log_probs.resize(count, 0.0);
        if space.discrete {
            self.discrete.resize(count, 0);
        } else {
            self.raw.resize(count * space.actions, 0.0);
        }
    }
}

#[derive(Default)]
struct Work {
    values: Vec<f32>,
    next_values: Vec<f32>,
    bootstrap_obs: Vec<f32>,
    bootstrap_rows: Vec<usize>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
    order: Vec<usize>,
    observations: Vec<f32>,
    probabilities: Vec<f32>,
    actor_delta: Vec<f32>,
    critic_delta: Vec<f32>,
    gradient: Vec<f32>,
}

pub(crate) type RngState = (u64, u128, u64, u128);
pub(crate) const METRICS: usize = 6;

pub(crate) struct Member {
    pub model: Model,
    pub weights: Vec<f32>,
    pub adam: Adam,
    pub rollout: Rollout,
    pub step_input: Vec<f32>,
    pub discrete_output: Vec<i64>,
    pub continuous_output: Vec<f32>,
    action_seed: u64,
    shuffle_seed: u64,
    action_rng: ChaCha8Rng,
    shuffle_rng: ChaCha8Rng,
    work: Work,
}

impl Member {
    pub(crate) fn new(model: Model, weights: Vec<f32>, adam: Adam, state: RngState) -> Self {
        let (action_seed, action_position, shuffle_seed, shuffle_position) = state;
        let mut action_rng = ChaCha8Rng::seed_from_u64(action_seed);
        let mut shuffle_rng = ChaCha8Rng::seed_from_u64(shuffle_seed);
        action_rng.set_word_pos(action_position);
        shuffle_rng.set_word_pos(shuffle_position);
        Self {
            model,
            weights,
            adam,
            rollout: Rollout::default(),
            step_input: Vec::new(),
            discrete_output: Vec::new(),
            continuous_output: Vec::new(),
            action_seed,
            shuffle_seed,
            action_rng,
            shuffle_rng,
            work: Work::default(),
        }
    }

    pub(crate) fn rng_state(&self) -> RngState {
        (
            self.action_seed,
            self.action_rng.get_word_pos(),
            self.shuffle_seed,
            self.shuffle_rng.get_word_pos(),
        )
    }

    pub(crate) fn actions(&mut self, space: &Space, rows: usize, sample_start: Option<usize>) {
        let input = if let Some(start) = sample_start {
            &self.rollout.obs[start * space.observations..(start + rows) * space.observations]
        } else {
            &self.step_input
        };
        let output = self.model.actor(&self.weights, input, rows);
        if space.discrete {
            self.discrete_output.resize(rows, 0);
            self.work.probabilities.resize(space.actions, 0.0);
        } else {
            self.continuous_output.resize(rows * space.actions, 0.0);
        }
        for (row, logits) in output.chunks_exact(output.len() / rows).enumerate() {
            let logits = &logits[..space.actions];
            if space.discrete {
                let action = if let Some(start) = sample_start {
                    let probabilities = &mut self.work.probabilities;
                    let (max, log_sum) = categorical(logits, probabilities);
                    let draw = self.action_rng.gen::<f32>();
                    let mut cumulative = 0.0;
                    let action = probabilities
                        .iter()
                        .position(|p| {
                            cumulative += p;
                            draw < cumulative
                        })
                        .unwrap_or(space.actions - 1);
                    self.rollout.discrete[start + row] = action as i64;
                    self.rollout.log_probs[start + row] = (logits[action] - max) - log_sum;
                    action
                } else {
                    logits
                        .iter()
                        .enumerate()
                        .fold((0, f32::NEG_INFINITY), |best, (i, &value)| {
                            if value > best.1 {
                                (i, value)
                            } else {
                                best
                            }
                        })
                        .0
                };
                self.discrete_output[row] = action as i64;
            } else {
                let log_std = &self.weights[self.weights.len() - space.actions..];
                let mut log_prob = 0.0;
                for (i, (&mean, &log_std)) in logits.iter().zip(log_std).enumerate() {
                    let raw = if let Some(start) = sample_start {
                        let deviation: f32 = self.action_rng.sample(StandardNormal);
                        let std = log_std.clamp(-5.0, 2.0);
                        let raw = mean + std.exp() * deviation;
                        self.rollout.raw[(start + row) * space.actions + i] = raw;
                        log_prob += gaussian(mean, raw, std, space.log_scale[i]).0;
                        raw
                    } else {
                        mean
                    };
                    self.continuous_output[row * space.actions + i] =
                        space.low[i] + (raw.tanh() + 1.0) * space.scale[i];
                }
                if let Some(start) = sample_start {
                    self.rollout.log_probs[start + row] = log_prob;
                }
            }
        }
    }

    pub(crate) fn train(&mut self, space: &Space, settings: &Settings) -> [f32; METRICS] {
        let count = settings.count().unwrap();
        let rollout = &self.rollout;
        self.work
            .prepare_returns(&mut self.model, &self.weights, rollout, space, settings);
        self.work.order.clear();
        self.work.order.extend(0..count);
        let mut totals = [0.0; METRICS];
        let mut updates = 0;
        for _ in 0..settings.epochs {
            self.work.order.shuffle(&mut self.shuffle_rng);
            for first in (0..count).step_by(settings.minibatch_size) {
                let mini = first..first.saturating_add(settings.minibatch_size).min(count);
                let terms = self.work.minibatch(
                    &mut self.model,
                    &self.weights,
                    rollout,
                    space,
                    settings,
                    mini,
                );
                let norm = self
                    .adam
                    .step(&mut self.weights, &self.work.gradient, settings);
                for (total, term) in totals.iter_mut().zip(terms.into_iter().chain([norm])) {
                    *total += term;
                }
                updates += 1;
            }
        }
        totals.map(|total| total / updates as f32)
    }
}

impl Work {
    fn prepare_returns(
        &mut self,
        model: &mut Model,
        weights: &[f32],
        rollout: &Rollout,
        space: &Space,
        settings: &Settings,
    ) {
        let count = rollout.rewards.len();
        let separate = model.separate_critic();
        let column = if separate { 0 } else { space.actions };
        let output = model.critic(weights, &rollout.obs, count);
        self.values.clear();
        self.values.extend(
            output
                .chunks_exact(output.len() / count)
                .map(|row| row[column]),
        );
        self.next_values.resize(count, 0.0);
        self.bootstrap_obs.clear();
        self.bootstrap_rows.clear();
        let copies = settings.env_copies;
        for i in 0..count {
            self.next_values[i] = 0.0;
            if !rollout.terminated[i] {
                if i + copies >= count || rollout.truncated[i] {
                    self.bootstrap_rows.push(i);
                    self.bootstrap_obs.extend_from_slice(
                        &rollout.final_obs[i * space.observations..(i + 1) * space.observations],
                    );
                } else {
                    self.next_values[i] = self.values[i + copies];
                }
            }
        }
        if !self.bootstrap_rows.is_empty() {
            let output = model.critic(weights, &self.bootstrap_obs, self.bootstrap_rows.len());
            let width = output.len() / self.bootstrap_rows.len();
            for (&index, value) in self.bootstrap_rows.iter().zip(output.chunks_exact(width)) {
                self.next_values[index] = value[column];
            }
        }
        self.advantages.resize(count, 0.0);
        self.returns.resize(count, 0.0);
        advantages(
            rollout,
            &self.values,
            &self.next_values,
            settings,
            &mut self.advantages,
            &mut self.returns,
        );
        normalize(&mut self.advantages);
    }

    fn minibatch(
        &mut self,
        model: &mut Model,
        weights: &[f32],
        rollout: &Rollout,
        space: &Space,
        settings: &Settings,
        mini: std::ops::Range<usize>,
    ) -> [f32; METRICS - 1] {
        let indices = &self.order[mini];
        let rows = indices.len();
        let reciprocal = 1.0 / rows as f32;
        self.observations.clear();
        for &i in indices {
            self.observations.extend_from_slice(
                &rollout.obs[i * space.observations..(i + 1) * space.observations],
            );
        }
        self.actor_delta.resize(rows * space.actions, 0.0);
        self.critic_delta.resize(rows, 0.0);
        self.gradient.resize(weights.len(), 0.0);
        self.gradient.fill(0.0);
        self.probabilities.resize(space.actions, 0.0);
        let log_std_count = if space.discrete { 0 } else { space.actions };
        let log_std_offset = weights.len() - log_std_count;
        let separate = model.separate_critic();
        let mut terms = [0.0; METRICS - 1];
        let logits = model.actor(weights, &self.observations, rows);
        let output_width = logits.len() / rows;
        for (row, &i) in indices.iter().enumerate() {
            let output = &logits[row * output_width..(row + 1) * output_width];
            let logits = &output[..space.actions];
            let delta = &mut self.actor_delta[row * space.actions..(row + 1) * space.actions];
            let (log_prob, entropy) = if space.discrete {
                let action = rollout.discrete[i] as usize;
                let (max, log_sum) = categorical(logits, &mut self.probabilities);
                let mut entropy = 0.0;
                for (&logit, &p) in logits.iter().zip(&self.probabilities) {
                    entropy -= p * ((logit - max) - log_sum);
                }
                let log_prob = (logits[action] - max) - log_sum;
                let policy = surrogate(
                    log_prob - rollout.log_probs[i],
                    self.advantages[i],
                    settings.clip,
                );
                for (j, &p) in self.probabilities.iter().enumerate() {
                    let entropy_grad = p * (((logits[j] - max) - log_sum) + entropy);
                    delta[j] = reciprocal
                        * (policy[1] * (f32::from(j == action) - p)
                            + settings.entropy_coefficient * entropy_grad);
                }
                (log_prob, entropy)
            } else {
                let log_std = &weights[log_std_offset..];
                let raw = &rollout.raw[i * space.actions..(i + 1) * space.actions];
                let mut log_prob = 0.0;
                let mut entropy = 0.0;
                for j in 0..space.actions {
                    let sigma = log_std[j].clamp(-5.0, 2.0);
                    let (log_p, z) = gaussian(logits[j], raw[j], sigma, space.log_scale[j]);
                    self.probabilities[j] = z;
                    log_prob += log_p;
                    entropy += sigma + 0.5 * (TAU.ln() + 1.0);
                }
                let policy = surrogate(
                    log_prob - rollout.log_probs[i],
                    self.advantages[i],
                    settings.clip,
                );
                for (j, &z) in self.probabilities.iter().enumerate() {
                    delta[j] = reciprocal * policy[1] * z * (-log_std[j].clamp(-5.0, 2.0)).exp();
                    if (-5.0..=2.0).contains(&log_std[j]) {
                        self.gradient[log_std_offset + j] +=
                            reciprocal * (policy[1] * (z * z - 1.0) - settings.entropy_coefficient);
                    }
                }
                (log_prob, entropy)
            };
            let policy = surrogate(
                log_prob - rollout.log_probs[i],
                self.advantages[i],
                settings.clip,
            );
            terms[0] += policy[0];
            terms[2] += entropy;
            terms[3] += policy[2];
            terms[4] += policy[3];
            if !separate {
                let error = output[space.actions] - self.returns[i];
                terms[1] += 0.5 * error * error;
                self.critic_delta[row] = reciprocal * settings.value_coefficient * error;
            }
        }
        if separate {
            let values = model.critic(weights, &self.observations, rows);
            for ((delta, &prediction), &i) in self.critic_delta.iter_mut().zip(values).zip(indices)
            {
                let error = prediction - self.returns[i];
                terms[1] += 0.5 * error * error;
                *delta = reciprocal * settings.value_coefficient * error;
            }
        }
        model.backward(
            weights,
            &self.observations,
            rows,
            &self.actor_delta,
            &self.critic_delta,
            &mut self.gradient,
        );
        terms.map(|total| total * reciprocal)
    }
}

fn advantages(
    rollout: &Rollout,
    values: &[f32],
    next: &[f32],
    settings: &Settings,
    advantages: &mut [f32],
    returns: &mut [f32],
) {
    let copies = settings.env_copies;
    for i in (0..values.len()).rev() {
        let terminal = rollout.terminated[i];
        let bootstrap = if terminal {
            0.0
        } else {
            settings.gamma * next[i]
        };
        let carry = if terminal || rollout.truncated[i] || i + copies >= values.len() {
            0.0
        } else {
            settings.gamma * settings.gae_lambda * advantages[i + copies]
        };
        advantages[i] = rollout.rewards[i] + bootstrap - values[i] + carry;
        returns[i] = advantages[i] + values[i];
    }
}

fn normalize(values: &mut [f32]) {
    let count = values.len() as f32;
    let mean = values.iter().sum::<f32>() / count;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / count;
    let scale = variance.sqrt() + 1e-8;
    values
        .iter_mut()
        .for_each(|value| *value = (*value - mean) / scale);
}

fn categorical(logits: &[f32], probabilities: &mut [f32]) -> (f32, f32) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for (p, logit) in probabilities.iter_mut().zip(logits) {
        *p = (logit - max).exp();
        sum += *p;
    }
    let inverse = sum.recip();
    probabilities.iter_mut().for_each(|p| *p *= inverse);
    (max, sum.ln())
}

fn gaussian(mean: f32, raw: f32, log_std: f32, log_scale: f32) -> (f32, f32) {
    let z = (raw - mean) / log_std.exp();
    let softplus = (-2.0 * raw).max(0.0) + (-2.0 * raw.abs()).exp().ln_1p();
    let jacobian = 2.0 * (LN_2 - raw - softplus);
    (
        -0.5 * (z * z + 2.0 * log_std + TAU.ln()) - jacobian - log_scale,
        z,
    )
}

/// Loss, derivative with respect to log probability, and the two PPO diagnostics.
fn surrogate(log_ratio: f32, advantage: f32, clip: f32) -> [f32; 4] {
    let ratio = log_ratio.exp();
    let bound = ratio.clamp(1.0 - clip, 1.0 + clip);
    let clipped = (advantage > 0.0 && ratio > bound) || (advantage < 0.0 && ratio < bound);
    let derivative = if clipped { 0.0 } else { -ratio * advantage };
    [
        -(ratio * advantage).min(bound * advantage),
        derivative,
        (ratio - 1.0) - log_ratio,
        f32::from((ratio - 1.0).abs() > clip),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> Settings {
        Settings {
            env_copies: 2,
            rollout_steps: 3,
            epochs: 1,
            minibatch_size: 2,
            learning_rate: 0.001,
            gamma: 0.9,
            gae_lambda: 0.8,
            clip: 0.2,
            value_coefficient: 0.5,
            entropy_coefficient: 0.01,
            max_gradient_norm: 0.5,
        }
    }

    fn close(actual: &[f32], expected: &[f32]) {
        for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            assert!((a - b).abs() < 2e-6, "{i}: {a} versus {b}");
        }
    }

    #[test]
    fn gae_bootstraps_truncation_and_cuts_recurrence_on_resets() {
        let rollout = Rollout {
            rewards: vec![1., 2., 3., 4., 5., 6.],
            terminated: vec![false, true, false, false, true, false],
            truncated: vec![false, true, true, false, false, false],
            ..Rollout::default()
        };
        let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let next = [0.3, 99., 0.8, 0.6, 99., 0.7];
        let mut adv = [0.; 6];
        let mut returns = [0.; 6];
        advantages(
            &rollout,
            &values,
            &next,
            &settings(),
            &mut adv,
            &mut returns,
        );
        close(&adv, &[3.6324, 1.8, 3.42, 8.4816, 4.5, 6.03]);
        close(&returns, &[3.7324, 2., 3.72, 8.8816, 5., 6.63]);
    }

    #[test]
    fn clipped_loss_and_derivative_match_reference() {
        close(&surrogate(1.5_f32.ln(), 2.0, 0.2)[..2], &[-2.4, 0.0]);
        close(&surrogate(0.5_f32.ln(), -2.0, 0.2)[..2], &[1.6, 0.0]);
        close(&surrogate(1.5_f32.ln(), -2.0, 0.2)[..2], &[3.0, 3.0]);
        close(&surrogate(0.9_f32.ln(), 2.0, 0.2)[..2], &[-1.8, -1.8]);
    }

    #[test]
    fn adam_keeps_per_member_moments_and_bias_correction() {
        let mut older = Adam::default();
        let mut newer = Adam::default();
        let (mut a, mut b) = ([0.0], [0.0]);
        older.step(&mut a, &[0.25], &settings());
        older.step(&mut a, &[-0.25], &settings());
        newer.step(&mut b, &[-0.25], &settings());
        assert_eq!((older.step, newer.step), (2, 1));
        close(&older.first, &[-0.0025]);
        close(&newer.first, &[-0.025]);
        close(&a, &[-0.00094733]);
        close(&b, &[0.00099996]);
    }
}
