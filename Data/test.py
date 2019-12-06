#!/usr/bin/env python3

from pprint import pprint

chicago = (-87, 41, 'IL', 'Windy City')
bloomington = (-89, 40, 'IL', 'Twin City')
indy = (-86, 40, 'IN', 'Naptown')
nodes = [ chicago, bloomington, indy ]
edges = [ 
    (chicago, bloomington, 'I-55'),
    (chicago, indy, 'I-65'),
    (indy, bloomington, 'I-74'),
]

adj = {}
for edge in edges:
    adj.setdefault(edge[0], set()).add(edge[1])
    adj.setdefault(edge[1], set()).add(edge[0])
pprint(adj)
