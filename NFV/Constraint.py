import pyomo.environ as pyo
model = pyo.ConcreteModel()

# =======================
# Sets
# =======================
model.V = pyo.Set()       # virtual nodes
model.Ev = pyo.Set()      # virtual links (u,v)
model.S = pyo.Set()       # substrate nodes
model.Es = pyo.Set()      # substrate links (i,j)

# =======================
# Parameters
# =======================
model.Cv = pyo.Param(model.V)                     # CPU demand of virtual nodes
model.Cs = pyo.Param(model.S)                     # CPU capacity of substrate nodes
model.Bv = pyo.Param(model.Ev)                    # bandwidth demand of virtual links
model.Bs = pyo.Param(model.Es)                    # bandwidth capacity of substrate links
model.Delay = pyo.Param(model.Es)                # delay of substrate links
model.MaxDelay = pyo.Param(model.Ev)             # delay constraint of virtual links
model.MaxDist = pyo.Param()   # maximum allowed distance D

# Virtual node target positions
model.Vx = pyo.Param(model.V)   # target x position for virtual nodes
model.Vy = pyo.Param(model.V)   # target y position for virtual nodes

# Substrate node real positions
model.Sx = pyo.Param(model.S)   # real x position for substrate nodes
model.Sy = pyo.Param(model.S)   # real y position for substrate nodes


# =======================
# Decision Variables
# =======================
model.x = pyo.Var(model.V, model.S, domain=pyo.Binary)                  # node mapping
model.y = pyo.Var(model.Ev, model.Es, domain=pyo.Binary)                # link mapping (path edges)

# =======================
# Constraints
# =======================

# 1. CPU capacity constraint (Eq. 18)
def cpu_constraint(model, s):
    return sum(model.x[v, s] * model.Cv[v] for v in model.V) <= model.Cs[s]
model.cpu_constraint = pyo.Constraint(model.S, rule=cpu_constraint)

# 2. Bandwidth capacity on substrate links (Eq. 19)
# Map virtual link uk to physical link mn; uv denotes uk and ij denotes mn.
def bw_constraint(model, i, j):
    return sum(model.y[(u, v), (i, j)] * model.Bv[(u, v)] for (u, v) in model.Ev) <= model.Bs[(i, j)]
model.bw_constraint = pyo.Constraint(model.Es, rule=bw_constraint)

# 3. Delay constraint (Eq. 20)
# Optional delay constraint.
def delay_constraint(model, u, v):
    return sum(model.y[(u, v), (i, j)] * model.Delay[(i, j)] for (i, j) in model.Es) <= model.MaxDelay[(u, v)]
model.delay_constraint = pyo.Constraint(model.Ev, rule=delay_constraint)

# 4. Unique mapping of virtual node -> substrate node (Eq. 21)
def one_node_map(model, v):
    return sum(model.x[v, s] for s in model.S) == 1
model.one_node_map = pyo.Constraint(model.V, rule=one_node_map)

# 5. (Optional) Only one virtual node per substrate node (Eq. 22, paper’s assumption)
def one_vn_per_sn(model, s):
    return sum(model.x[v, s] for v in model.V) <= 1
model.one_vn_per_sn = pyo.Constraint(model.S, rule=one_vn_per_sn)

# 6. Each virtual link maps to at least one substrate path (Eq. 23)
def link_exist(model, u, v):
    return sum(model.y[(u, v), (i, j)] for (i, j) in model.Es) >= 1
model.link_exist = pyo.Constraint(model.Ev, rule=link_exist)

# 7. Flow conservation constraint (Eq. 24)
def flow_conservation(model, u, v, s):
    return (sum(model.y[(u, v), (s, j)] for j in model.S if (s, j) in model.Es) -
            sum(model.y[(u, v), (j, s)] for j in model.S if (j, s) in model.Es)
            ==
            model.x[u, s] - model.x[v, s])
model.flow_conservation =pyo. Constraint(model.Ev, model.S, rule=flow_conservation)

def geo_constraint(model, v, s):
    return ((model.Vx[v] - model.Sx[s])**2 +
            (model.Vy[v] - model.Sy[s])**2
           ) <= model.MaxDist**2
model.geo_constraint = pyo.Constraint(model.V, model.S, rule=geo_constraint)