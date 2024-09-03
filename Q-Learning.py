import numpy as np

rwds = np.array([
    [-1, 0, -1, -1, 0, -1],
    [0, -1, -1, 0, -1, 100],
    [-1, -1, -1, 0, -1, -1],
    [-1, 0, 0, -1, 0, -1],
    [0, -1, -1, 0, -1, 100],
    [-1, 0, -1, -1, 0, 100]
])

Q_vals = np.zeros_like(rwds)

alpha_val = 0.2
gamma_val = 0.8
num_eps = 1000

for _ in range(num_eps):
    cur_st = np.random.randint(0, rwds.shape[0])

    while True:
        poss_acts = np.where(rwds[cur_st] >= 0)[0]
        act = np.random.choice(poss_acts)

        nxt_st = act

        Q_vals[cur_st, act] = (1 - alpha_val) * Q_vals[cur_st, act] + alpha_val * (rwds[cur_st, act] + gamma_val * np.max(Q_vals[nxt_st]))

        cur_st = nxt_st

        if cur_st == 5:
            break

print("Q MATRIX:")
print(Q_vals)

rooms = ['A', 'B', 'C', 'D', 'E', 'F']
sol_paths = {}
for st in range(rwds.shape[0]):
    cur_st = st
    path = [rooms[cur_st]]
    while cur_st != 5:  
        nxt_act = np.argmax(Q_vals[cur_st])
        nxt_st = nxt_act
        path.append(rooms[nxt_st])
        cur_st = nxt_st
    sol_paths[rooms[st]] = path

print("PATHS TO GOAL:")
for room, path in sol_paths.items():
    print(f"Room {room}:", end=" ")
    print("->".join(path))
