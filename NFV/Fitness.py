# Total node-mapping cost.
def total_cost_count(node_cpu, link_bw, w_node=0.5, w_link=0.5):
    cpu_cost = sum(node_cpu)
    bw_cost = sum(link_bw)

    return w_node * cpu_cost + w_link * bw_cost

# Total link-mapping cost.
def hop_sum(physical_paths):

    total_hops = 0

    for vlink, path in physical_paths.items():
        # hops = number of edges = len(path)-1
        hops = max(len(path) - 1, 0)
        total_hops += hops

    return total_hops


def vne_fitness(cost, hop_sum, w_c=0.5, w_h=0.5):

    # avoid division by zero
    if cost == 0:
        cost = 1e-6
    if hop_sum == 0:
        hop_sum = 1e-6

    fitness = (1.0 / cost) * w_c + (1.0 / hop_sum) * w_h
    return fitness


# ===== Example usage =====
if __name__ == "__main__":
    # Example: VNE solution quality indicators
    total_cost = 120     # resource consumption (lower better)
    total_hops = 10      # hops (lower better)

    # Compute fitness
    score = vne_fitness(total_cost, total_hops, w_c=0.6, w_h=0.4)

    print(f"Fitness score: {score:.6f}")