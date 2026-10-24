import numpy as np
import pymysql
from NFV.db_config import connect_database

from NFV.network_flow.TM_generator import TrafficMatrixCalculator

conn = connect_database()
# Five network domains.
domain_nodes = [19, 0, 1, 2, 5]
#domain_nodes = [9, 4, 10, 11, 12]
#domain_nodes = [3, 7, 8, 13, 14]
#domain_nodes = [16, 15, 17, 18, 21]
#domain_nodes = [6, 20, 22, 23, 24]

tm_calculator = TrafficMatrixCalculator(conn)

tm_result = tm_calculator.compute_tm(domain_nodes)

TM = tm_result["TM"]
nodes = tm_result["nodes"]

print("Nodes:", nodes)
print("Traffic Matrix:\n", TM)

H = tm_calculator.compute_network_entropy(TM, log_base=np.e)  # nats
print("Network Entropy H =", H)