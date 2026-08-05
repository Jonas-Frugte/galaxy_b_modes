import numpy as np
import numpy.random as rand
import itertools
import matplotlib.pyplot as plt

n_side = 30
n_halos = n_side**3
rmax = 2
ang_spec_max = 0.01 * np.pi
pos_obs = np.array([0.5, 0.5, 0.5])

# coords_1d = np.linspace(0, 1, n_side)
rng = np.random.default_rng(0)
coords_3d = rng.uniform(0, 1, size=(n_halos, 3))

jitter = False
jitter_amp = 0.01
if jitter:
    coords_3d += rand.uniform(-jitter_amp, jitter_amp, size=coords_3d.shape)

def delta_ang(v1s, v2s):
    return np.arccos(np.clip(np.einsum('ij,ij->i', v1s, v2s) / (np.linalg.norm(v1s, axis=1) * np.linalg.norm(v2s, axis=1)), -1, 1))

replication_steps = 4

possible_steps = np.array([
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1]
])

special_dirs = np.array([
    [0, 0, 1],
    [0, 0, -1],
    [0, 1, 0],
    [0, -1, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [1, 1, 0],
    [-1, -1, 0],
    [1, -1, 0],
    [-1, 1, 0],
    [1, 0, 1],
    [-1, 0, -1],
    [1, 0, -1],
    [-1, 0, 1],
    [0, 1, 1],
    [0, -1, -1],
    [0, 1, -1],
    [0, -1, 1],
    [1, 1, 1],
    [-1, -1, -1],
    [1, 1, -1],
    [-1, -1, 1],
    [1, -1, 1],
    [-1, 1, -1],
    [1, -1, -1],
    [-1, 1, 1]
])

routes_not_unique = np.concatenate([np.sum(np.array(list(itertools.combinations_with_replacement(possible_steps, i))), axis=1) for i in range(1, replication_steps+1)])
routes = np.unique(routes_not_unique, axis=0)

boxes = [coords_3d + route for route in routes]
if replication_steps < 2:
    boxes.append(coords_3d)

rs = np.array([np.linalg.norm(box - pos_obs, axis=1) for box in boxes])
inside = rs < rmax
angs_special = np.array([[delta_ang((box - pos_obs)[:, :], special_dir[None, :]) for special_dir in special_dirs] for box in boxes])
angs_special_min = np.min(angs_special, axis=1)
is_inside_ang_spec = angs_special_min < ang_spec_max

angs_total = []
for i, j in list(itertools.combinations(range(len(boxes)), 2)):
    box_1, box_2 = boxes[i], boxes[j]
    both_inside_r = inside[i] & inside[j]
    both_inside_ang = is_inside_ang_spec[i] & is_inside_ang_spec[j]
    both_inside = both_inside_r & both_inside_ang
    if not both_inside.any():
        continue
    angs = delta_ang((box_1 - pos_obs)[both_inside], (box_2 - pos_obs)[both_inside])
    angs_total.extend(angs)

angs_total = np.array(angs_total)
plt.hist(np.clip(np.pi/angs_total, 0, 4000), bins=100, density=True)
plt.show()
