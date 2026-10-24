import pymysql
if __package__:
    from .db_config import connect_database
else:
    from db_config import connect_database

conn = connect_database()

links = [
    (1,2),(1,4),(1,6),(1,12),(1,13),(1,22),(1,25),
    (2,3),(2,4),(2,5),(2,12),(2,13),(2,15),(2,17),(2,19),(2,20),(2,21),(2,22),
    (3,5),(3,7),(3,8),(3,11),(3,13),(3,19),(3,23),
    (4,6),(4,9),(4,12),(4,18),(4,19),(4,21),(4,25),
    (5,6),(5,8),(5,9),(5,13),(5,15),(5,17),(5,20),(5,25),
    (6,7),(6,8),(6,9),(6,13),(6,15),(6,19),(6,20),
    (7,8),(7,10),(7,14),(7,17),(7,21),(7,22),(7,24),
    (8,9),(8,12),(8,13),(8,15),(8,18),(8,19),(8,20),(8,21),(8,22),(8,23),(8,25),
    (9,11),(9,12),(9,19),(9,20),(9,21),
    (10,14),(10,21),(10,23),(10,24),
    (11,12),(11,13),(11,19),(11,20),(11,22),
    (12,13),(12,15),(12,16),(12,21),(12,23),(12,25),
    (13,16),(13,19),(13,20),(13,23),(13,24),
    (14,23),(14,24),
    (15,16),(15,17),(15,19),(15,22),
    (16,19),(16,21),(16,23),
    (17,22),(17,23),
    (18,20),(18,21),(18,24),
    (19,20),(19,21),(19,22),(19,23),(19,24),
    (20,21),(20,23),
    (21,22),(21,25)
]

port_count = {i: 2 for i in range(1, 26)}

cursor = conn.cursor()

for u, v in links:
    pu = port_count[u]
    pv = port_count[v]

    mac_u = f'00:00:0a:{u:02d}:{pu:02d}:01'
    mac_v = f'00:00:0a:{v:02d}:{pv:02d}:01'

    # Mininet node IDs: s1 through s25.
    # Database node IDs: 0 through 24.
    db_u = u - 1
    db_v = v - 1

    # Forward link: u -> v.
    # Traffic leaves s_u, so egress_port is pu.
    # The next-hop MAC is the peer s_v interface MAC.
    cursor.execute("""
        UPDATE link_single_table
        SET egress_port = %s,
            next_hop_mac = %s
        WHERE source_node = %s
          AND destination_node = %s
    """, (pu, mac_v, db_u, db_v))

    # Reverse link: v -> u.
    # Traffic leaves s_v, so egress_port is pv.
    # The next-hop MAC is the peer s_u interface MAC.
    cursor.execute("""
        UPDATE link_single_table
        SET egress_port = %s,
            next_hop_mac = %s
        WHERE source_node = %s
          AND destination_node = %s
    """, (pv, mac_u, db_v, db_u))

    port_count[u] += 1
    port_count[v] += 1

conn.commit()
cursor.close()
conn.close()

print("egress_port 和 next_hop_mac 已写入数据库")