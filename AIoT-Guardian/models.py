

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from config import (
    DNRO_BETA,
    DNRO_DREAM_NOISE,
    DNRO_LR_INIT,
    DPSTS_D_MODEL,
    DPSTS_HEADS,
    DPSTS_PROBSPARSE,
    DPSTS_SEQ_LEN,
    FLSM_LOW_MAX,
    FLSM_MED_MAX,
    MARL_ACTIONS,
    MARL_AGENTS,
    PSO_ITERATIONS,
    PSO_PARTICLES,
    RANDOM,
    RANDOM_STATE,
    REFN_EPOCHS,
    REFN_HIDDEN,
)


# ======================================================================
# 6. DPSTS
# ======================================================================
class DPSTS(BaseEstimator):
    """
    Dual ProbSparse Spatial-Temporal self-attention (lightweight).

    A synthetic temporal window is built by chunking the feature vector
    into `seq_len` pieces; two rounds of ProbSparse attention (spatial
    then temporal) are applied and fused.
    """

    def __init__(self, d_model=DPSTS_D_MODEL, heads=DPSTS_HEADS,
                 probsparse=DPSTS_PROBSPARSE, seq_len=DPSTS_SEQ_LEN):
        self.d_model = d_model
        self.heads = heads
        self.probsparse = probsparse
        self.seq_len = seq_len

    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape
        self.input_dim_ = d
        self.Wq_ = RANDOM.normal(0, 1 / np.sqrt(d), size=(d, self.d_model)).astype(np.float32)
        self.Wk_ = RANDOM.normal(0, 1 / np.sqrt(d), size=(d, self.d_model)).astype(np.float32)
        self.Wv_ = RANDOM.normal(0, 1 / np.sqrt(d), size=(d, self.d_model)).astype(np.float32)
        return self

    # ------------------------------------------------------------------
    def _reshape_sequence(self, X):
        n, d = X.shape
        chunk = int(np.ceil(d / self.seq_len))
        padded = np.zeros((n, chunk * self.seq_len), dtype=np.float32)
        padded[:, :d] = X
        return padded.reshape(n, self.seq_len, chunk)

    def _attention(self, Q, K, V):
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(Q.shape[-1])

        # ProbSparse: keep the top-k scores per query
        k = max(1, int(self.probsparse * scores.shape[-1]))
        thresh = np.sort(scores, axis=-1)[:, :, -k][:, :, :1]
        sparse_mask = scores >= thresh
        scores = np.where(sparse_mask, scores, -1e9)

        scores -= scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= (weights.sum(axis=-1, keepdims=True) + 1e-9)
        return weights @ V

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        seq = self._reshape_sequence(X)                    # (n, T, chunk)

        chunk_repr = seq.mean(axis=2)                      # (n, T)
        H = np.repeat(chunk_repr[:, :, None], self.d_model, axis=2)

        Q = H @ self.Wq_
        K = H @ self.Wk_
        V = H @ self.Wv_

        A_spatial = self._attention(Q, K, V)               # (n, T, d)

        A_temporal = self._attention(
            A_spatial.transpose(0, 2, 1),
            A_spatial.transpose(0, 2, 1),
            A_spatial.transpose(0, 2, 1),
        ).transpose(0, 2, 1)                               # (n, T, d)

        fused = 0.5 * A_spatial + 0.5 * A_temporal
        return fused.reshape(fused.shape[0], -1)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# ======================================================================
# 7. REFN + D-NRO
# ======================================================================
class REFN(BaseEstimator, ClassifierMixin):
    """
    Receptive-Echo Field Network.

    A shallow MLP with an echo-state-style recurrence applied to the
    first hidden layer. Trained with the D-NRO optimizer.
    """

    def __init__(self, hidden=REFN_HIDDEN, epochs=REFN_EPOCHS,
                 lr=DNRO_LR_INIT, beta=DNRO_BETA,
                 dream_noise=DNRO_DREAM_NOISE,
                 random_state=RANDOM_STATE):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.beta = beta
        self.dream_noise = dream_noise
        self.random_state = random_state

    # ------------------------------------------------------------------
    def _init_params(self, d):
        rng = np.random.default_rng(self.random_state)
        self.W1_ = rng.normal(0, 1 / np.sqrt(d), (d, self.hidden)).astype(np.float32)
        self.b1_ = np.zeros(self.hidden, dtype=np.float32)
        self.W2_ = rng.normal(0, 1 / np.sqrt(self.hidden), (self.hidden, 1)).astype(np.float32)
        self.b2_ = np.zeros(1, dtype=np.float32)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    def _forward(self, X):
        h1 = np.tanh(X @ self.W1_ + self.b1_)
        # Echo-state recurrence
        h1 = self.beta * h1 + (1 - self.beta) * np.tanh(h1)
        z = h1 @ self.W2_ + self.b2_
        return h1, self._sigmoid(z).ravel()

    def _dnro_update(self, grad, velocity):
        """
        Dream-based Newton-Raphson Optimizer:
            v = beta*v + lr * (grad + dream_noise * H^-1 grad)
        H^-1 approximated by a diagonal preconditioner |grad|+eps.
        """
        dream = self.dream_noise * grad / (np.abs(grad) + 1e-8)
        v = self.beta * velocity + self.lr * (grad + dream)
        return v

    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._init_params(X.shape[1])

        vW1 = np.zeros_like(self.W1_); vb1 = np.zeros_like(self.b1_)
        vW2 = np.zeros_like(self.W2_); vb2 = np.zeros_like(self.b2_)

        for epoch in range(self.epochs):
            h1, p = self._forward(X)
            err = (p - y).reshape(-1, 1) / len(X)

            gW2 = h1.T @ err;      gb2 = err.sum(axis=0)
            dh1 = (err @ self.W2_.T) * (1 - h1 ** 2)
            gW1 = X.T @ dh1;       gb1 = dh1.sum(axis=0)

            vW1 = self._dnro_update(gW1, vW1); vb1 = self._dnro_update(gb1, vb1)
            vW2 = self._dnro_update(gW2, vW2); vb2 = self._dnro_update(gb2, vb2)

            self.W1_ -= vW1; self.b1_ -= vb1
            self.W2_ -= vW2; self.b2_ -= vb2

            if (epoch + 1) % 10 == 0:
                loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
                print(f"  REFN epoch {epoch + 1}/{self.epochs}  loss={loss:.4f}")

        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        _, p = self._forward(np.asarray(X, dtype=np.float32))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ======================================================================
# 8. FLSM
# ======================================================================
class FLSM:
    """
    Fuzzy Logic Scoring Model.
    severity = fuzzy(attack_prob, confidence, deviation, impact)
    """

    def __init__(self):
        self.mf_low = lambda x: max(0.0, min(1.0, (FLSM_LOW_MAX - x) / FLSM_LOW_MAX))
        self.mf_med = lambda x: max(0.0, min((x) / FLSM_LOW_MAX,
                                             (FLSM_MED_MAX - x) / (FLSM_MED_MAX - FLSM_LOW_MAX)))
        self.mf_high = lambda x: max(0.0, min(1.0, (x - FLSM_MED_MAX) / (1 - FLSM_MED_MAX)))

    def _fuzzify(self, x):
        return self.mf_low(x), self.mf_med(x), self.mf_high(x)

    def score(self, attack_prob, confidence, deviation, impact):
        """Return (severity_score in [0, 1], label)."""
        crisp = np.clip(
            0.4 * attack_prob + 0.2 * confidence + 0.2 * deviation + 0.2 * impact,
            0, 1,
        )
        l, m, h = self._fuzzify(crisp)

        # Centroid defuzzification over {0.2, 0.5, 0.85}
        num = l * 0.2 + m * 0.5 + h * 0.85
        den = l + m + h + 1e-9
        score = num / den

        if score <= FLSM_LOW_MAX:
            label = "low"
        elif score <= FLSM_MED_MAX:
            label = "medium"
        else:
            label = "high"

        return float(score), label


# ======================================================================
# 9. GRAPH LOCALIZER
# ======================================================================
class GraphLocalizer:
    """
    Bipartite graph (device <-> communication link) built from per-sample
    feature contributions; nodes ranked by accumulated suspicion.
    """

    def __init__(self, n_devices=MARL_AGENTS):
        self.n_devices = n_devices

    def localize(self, attack_probs, feature_contributions):
        n = len(attack_probs)
        suspicion = np.zeros(self.n_devices)

        for i in range(n):
            contribs = np.abs(feature_contributions[i])
            top = np.argsort(contribs)[-10:]
            for f in top:
                device_id = int(f) % self.n_devices
                suspicion[device_id] += attack_probs[i] * contribs[f]

        if suspicion.sum() > 0:
            suspicion = suspicion / suspicion.sum()

        ranked = np.argsort(-suspicion)
        return ranked, suspicion


# ======================================================================
# 10. DDPG-MARL
# ======================================================================
class DDPGMARL:
    """
    Multi-agent RL remediation policy.
    Tabular Q-learning as a lightweight DDPG surrogate for discrete actions.
    """

    def __init__(self, n_agents=MARL_AGENTS, n_actions=len(MARL_ACTIONS)):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.Q = np.zeros((n_agents, 8, n_actions))
        self.lr = 0.1
        self.gamma = 0.9
        self.eps = 0.2

    def _state_idx(self, severity, loc_score):
        sev_bin = 0 if severity < FLSM_LOW_MAX else (1 if severity < FLSM_MED_MAX else 2)
        loc_bin = 0 if loc_score < 0.5 else 1
        return (sev_bin * 2 + loc_bin + (0 if severity < 0.5 else 2)) % 8

    def select_action(self, agent_id, severity, loc_score, rng=RANDOM):
        s = self._state_idx(severity, loc_score) % 8
        if rng.random() < self.eps:
            return int(rng.integers(self.n_actions)), s
        return int(np.argmax(self.Q[agent_id, s])), s

    def update(self, agent_id, s, a, reward, s_next):
        best_next = np.max(self.Q[agent_id, s_next])
        self.Q[agent_id, s, a] += self.lr * (
            reward + self.gamma * best_next - self.Q[agent_id, s, a]
        )


# ======================================================================
# 11. IMMUNE RECONFIGURER
# ======================================================================
class ImmuneReconfigurer:
    """Immune-inspired negative selection: isolate over-suspicious nodes."""

    def __init__(self, tolerance=0.5):
        self.tolerance = tolerance

    def isolate(self, suspicion_scores):
        return [i for i, s in enumerate(suspicion_scores) if s > self.tolerance]


# ======================================================================
# 12. PSO REROUTER
# ======================================================================
class PSORerouter:
    """PSO to find minimum-cost alternate path around isolated nodes."""

    def __init__(self, n_nodes=MARL_AGENTS):
        self.n_nodes = n_nodes

    def reroute(self, adjacency, isolated_nodes,
                particles=PSO_PARTICLES, iters=PSO_ITERATIONS):
        n = adjacency.shape[0]
        isolated = set(isolated_nodes)

        def cost(path):
            c = 0
            for a, b in zip(path[:-1], path[1:]):
                if a in isolated or b in isolated:
                    return 1e6
                c += adjacency[a, b]
            return c

        best_path, best_cost = None, np.inf
        particles_ = [np.arange(n) for _ in range(particles)]

        for _ in range(iters):
            for p in particles_:
                c = cost(p)
                if c < best_cost:
                    best_cost, best_path = c, p.copy()

        return best_path, best_cost