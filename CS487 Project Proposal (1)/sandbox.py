import numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
np.random.seed(0)

## baseline distribution P0 
P0 = np.array([0.20, 0.15, 0.13, 0.10, 0.09,
               0.08, 0.07, 0.07, 0.06, 0.05])

## offline preference pairs 
def sample_pairs(dist, n=50_000):
    pos, neg = [], []
    for _ in range(n):
        a, b = np.random.choice(10, 2, p=dist)
        pos.append(a if dist[a] >= dist[b] else b)
        neg.append(b if dist[a] >= dist[b] else a)
    return np.array(pos), np.array(neg)

pos_pairs, neg_pairs = sample_pairs(P0)

def train_reward_weighted(pos, neg, weights):
    X = np.eye(10)[np.concatenate([pos, neg])]
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    w = np.concatenate([weights, np.ones(len(neg))])      # neg weight = 1
    clf = LogisticRegression(solver="lbfgs").fit(X, y, sample_weight=w)
    return clf

# initial RM: uniform weights (=1)
reward_model = train_reward_weighted(pos_pairs, neg_pairs,
                                     np.ones(len(pos_pairs)))

BATCH   = 10_000   # rollout size
ALPHA   = 0.01     # policy step size
KL_THR  = 0.05     # refresh threshold (REFERENCE KL)
STEPS   = 200


def ppo_update(P_old, rollout, reward, lr=ALPHA):
    Δ = np.bincount(rollout, weights=reward, minlength=10)
    Δ = lr * Δ / Δ.sum()
    P_new = P_old * (1 - lr) + Δ
    return P_new / P_new.sum()

def kl_batch(Pn, Po, samp):       # step KL
    return np.mean(np.log(Pn[samp]) - np.log(Po[samp]))

def kl_reference(Pn):
    return np.sum(Pn * np.log(Pn / P0))

## training
P_policy = P0.copy()
logs = {"kl_step": [], "kl_ref": [], "mean_r": []}

for t in range(STEPS):
    # 5-1  roll-out
    rollout = np.random.choice(10, BATCH, p=P_policy)
    rewards = reward_model.predict_proba(np.eye(10)[rollout])[:, 1]

    # 5-2  policy update
    P_new = ppo_update(P_policy, rollout, rewards)
    step_kl = kl_batch(P_new, P_policy, rollout)
    ref_kl  = kl_reference(P_new)

    logs["kl_step"].append(step_kl)
    logs["kl_ref"].append(ref_kl)
    logs["mean_r"].append(rewards.mean())
    P_policy = P_new.copy()

    ## Re-weigthing
    if ref_kl > KL_THR:
        rho      = P_policy / P0              # density ratio vector
        w_pos    = rho[pos_pairs] / rho[pos_pairs].mean()   # normalise
        reward_model = train_reward_weighted(pos_pairs, neg_pairs, w_pos)
        print(f"[iter {t}] RM re-weighted ρ̄={rho.mean():.2f}  ref-KL={ref_kl:.3f}")

## visualisation
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].plot(logs["kl_step"]); ax[0].set_title("Step KL  (new‖old)")
ax[1].plot(logs["kl_ref"]);  ax[1].set_title("Reference KL  (new‖P0)")
ax[2].plot(logs["mean_r"]);  ax[2].set_title("Mean reward")
for a in ax: a.set_xlabel("iteration")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 4))
plt.plot(P0,       "k-o", label="baseline P0")
plt.plot(P_policy, "b--o",label="final policy")
plt.xticks(range(10), [f"x{i}" for i in range(10)])
plt.ylabel("probability"); plt.legend(); plt.title("Distributions"); plt.show()
