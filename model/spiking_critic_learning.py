"""spiking_critic_learning.py — Python-side, centered-presynaptic,
delta-gated update for the spiking critic's cortex->L1 connection,
replacing native `critic_dopsyn` STDP there. Same
diagnosis and fix pattern as `batched_actor_learning.py`'s BatchedD1Learning
(uncentered eligibility collapses -- see that module's own docstring),
applied to the critic's one plastic
layer instead of the actor's cortex->D1.

Structurally this makes the spiking critic the direct analogue of
`python_critic.py`'s own linear TD(0) critic (`V = w @ phi(s)`, `w +=
alpha*delta*phi(s)`) -- "features" here are real cortex spike counts
(centered against each cortex neuron's own running mean, same convention
as the actor's presynaptic centering) instead of a hand-built `phi(obs)`,
and `V(s)` is the critic's L1 population RATE over the state's own
activation window (reusing `Critic.diagnose_value_signal()`/
`Base.rate_in_window()`, not reimplemented here) instead of a dot
product.

One-step lag, not a shortcut: `python_critic.py` gets V(next_obs) "for
free" (phi is a deterministic, instant function of obs). A spiking
critic's V(s') requires actually simulating cortex's response to s' --
which happens anyway, one control step later, as the NEXT window's
activation. So each step's weight update is deferred by exactly one
control step: step t's (x_counts, v_s) is cached; once step t+1's v_s
becomes available (serving as v_next for the transition at t), delta_t =
r_t + gamma*v_next - v_s_t is computed and applied using step t's own
cortex activity. A terminal step (done=True) resolves immediately with
v_next=0, matching the standard semi-gradient TD(0) terminal-state
convention -- no deferred update carries past an episode boundary.

Delta centering: centering the presynaptic term alone isn't enough if
`delta` itself has a persistent mean bias (this task's reward is
negative-skewed). A consistently negative delta, times any presynaptic
term, reliably drags the mean weight in one direction regardless of
real state-dependence. Fix: track an EMA baseline of delta itself
(delta_bar, same alpha convention as x_bar) and use (delta-delta_bar)
in the update instead of raw delta -- same idea as batched_actor_
learning.py's own `center_nb_per_batch` for the actor's dopamine
signal, applied here to the critic's TD error. Centers against the
PRIOR delta_bar (computed before this delta folds in), same convention
as x_bar's own "baseline updates AFTER this step's deviation is
computed" rule.
"""
import numpy as np
import nest


class SpikingCriticLearning:
    def __init__(self, cortex, critic_l1, synapse_model, lr: float,
                 xbar_alpha: float = 0.05, wmin: float = 5.0, wmax: float = 15.0,
                 gamma: float = 0.97, delta_bar_alpha: float = 0.05,
                 center_delta: bool = True):
        self.synapse_model = synapse_model
        self.lr = lr
        self.xbar_alpha = xbar_alpha
        self.wmin = wmin
        self.wmax = wmax
        self.gamma = gamma
        self.delta_bar_alpha = delta_bar_alpha
        self.delta_bar = 0.0
        self.center_delta = center_delta
        self.n_src = len(cortex)
        self.n_tgt = len(critic_l1)
        # Full (n_src, n_tgt) weight matrix -- only entries at
        # (conn_src, conn_tgt) correspond to real connections; the rest
        # are inert (fixed_indegree, not all_to_all). Same convention as
        # the CartPole/arm actor's own W_d1/W_ch tracking.
        self.W = np.zeros((self.n_src, self.n_tgt))
        self.x_bar = np.zeros(self.n_src)
        self._pending = None   # (x_counts, x_bar_at_time, v_s, reward)
        self.attach(cortex, critic_l1)

    def attach(self, cortex, critic_l1):
        """(Re)acquire connection handles and id maps -- call again after
        any nest.ResetKernel() rebuild (connection handles go stale, pop
        ids renumber). self.W/self.x_bar are plain numpy arrays owned by
        this object, unaffected by a kernel reset -- only the
        (conn_src, conn_tgt) entries are re-synced here from the ACTUAL
        NEST weights (NestCheckpointer's own "input_critic_l1" projection
        tracking already restored those onto the rebuilt connections;
        this just re-reads them into this object's cache, not re-derives
        anything)."""
        self.cortex = cortex
        self.l1 = critic_l1
        src_ids = cortex.tolist()
        tgt_ids = critic_l1.tolist()
        self._src_idx = {nid: i for i, nid in enumerate(src_ids)}
        self._tgt_idx = {nid: i for i, nid in enumerate(tgt_ids)}
        self.conns = nest.GetConnections(cortex, critic_l1, synapse_model=self.synapse_model)
        self.conn_src = np.array([self._src_idx[int(s)] for s in self.conns.get("source")])
        self.conn_tgt = np.array([self._tgt_idx[int(t)] for t in self.conns.get("target")])
        self.W[self.conn_src, self.conn_tgt] = self.conns.get("weight")
        self._pending = None   # any pending delayed update is now stale (new episode/rebuild)

    def step(self, cortex_rec, t0: float, t1: float, v_s: float, reward: float, done: bool):
        """Call once per control step, AFTER the step's spikes have
        occurred (t0,t1 = that step's activation window) and v_s =
        net.critic.diagnose_value_signal(t0, t1) has already been
        computed by the caller (reused, not recomputed here).

        Returns the (centered) delta actually applied this call, or None
        if nothing resolved (first step of an episode, before any
        transition is pending) -- also reused as the actor's training
        signal when actor_batched_d1_
        signal_source="delta_spiking" (main.py), one control step lagged
        behind the action that caused it, same lag the weight update
        itself already has (see this module's own docstring)."""
        x_counts = self._counts(cortex_rec, self._src_idx, self.n_src, t0, t1)
        x_bar_now = self.x_bar.copy()   # baseline BEFORE this step's counts fold in
        resolved_delta = None

        if self._pending is not None:
            prev_x, prev_xbar, prev_v, prev_reward = self._pending
            delta = prev_reward + self.gamma * v_s - prev_v
            resolved_delta = self._center_delta(delta)
            self._apply(prev_x, prev_xbar, resolved_delta)
            self._pending = None

        if done:
            delta = reward - v_s
            resolved_delta = self._center_delta(delta)
            self._apply(x_counts, x_bar_now, resolved_delta)
        else:
            self._pending = (x_counts, x_bar_now, v_s, reward)

        self.x_bar = (1.0 - self.xbar_alpha) * self.x_bar + self.xbar_alpha * x_counts
        return resolved_delta

    def _center_delta(self, delta):
        """Centers against the PRIOR delta_bar (computed before this delta
        folds in), same convention as x_bar's own baseline update.
        center_delta=False (P1.2b Arm B) skips this -- raw delta used."""
        if not self.center_delta:
            return delta
        centered_delta = delta - self.delta_bar
        self.delta_bar = (1.0 - self.delta_bar_alpha) * self.delta_bar + self.delta_bar_alpha * delta
        return centered_delta

    def _apply(self, x_counts, x_bar, delta):
        centered = x_counts - x_bar
        self.W += self.lr * delta * centered[:, None]
        np.clip(self.W, self.wmin, self.wmax, out=self.W)
        self.conns.set(weight=self.W[self.conn_src, self.conn_tgt].tolist())

    @staticmethod
    def _counts(rec, id_to_idx, n, t0, t1):
        events = nest.GetStatus(rec, keys="events")[0]
        times = np.asarray(events["times"])
        senders = np.asarray(events["senders"])
        counts = np.zeros(n)
        if len(times) == 0:
            return counts
        mask = (times >= t0) & (times < t1)
        senders = senders[mask]
        if len(senders) == 0:
            return counts
        idxs = np.fromiter((id_to_idx[s] for s in senders), dtype=int, count=len(senders))
        np.add.at(counts, idxs, 1)
        return counts
