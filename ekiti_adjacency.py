"""
Adjacency matrix for the towns lying inside the Ekiti State boundary
(the red dotted line on the map).

Adjacency = a direct road link / shared neighbourhood on the map,
with no other listed town in between. The graph is undirected, so the
matrix is symmetric with a zero diagonal.
"""

import numpy as np

# Row / column labels (index order is fixed by this list)
labels = [
    "Otun Ekiti",    # 0
    "Ido Ekiti",     # 1
    "Ijero",         # 2
    "Oke-Mesi",      # 3
    "Efon Alaaye",   # 4
    "Ikogosi",       # 5
    "Aramoko",       # 6
    "Iyin",          # 7
    "Ifaki",         # 8
    "Iworoko",       # 9
    "Oye",           # 10
    "Ikole",         # 11
    "Omuo-Oke",      # 12
    "Ado Ekiti",     # 13
    "Ikere-Ekiti",   # 14
    "Ise",           # 15
]

edges = [
    ("Otun Ekiti", "Ido Ekiti"),
    ("Otun Ekiti", "Ifaki"),
    ("Ido Ekiti", "Ijero"),
    ("Ido Ekiti", "Ifaki"),
    ("Ijero", "Oke-Mesi"),
    ("Ijero", "Aramoko"),
    ("Oke-Mesi", "Efon Alaaye"),
    ("Oke-Mesi", "Ikogosi"),
    ("Efon Alaaye", "Ikogosi"),
    ("Ikogosi", "Aramoko"),
    ("Aramoko", "Iyin"),
    ("Aramoko", "Ado Ekiti"),
    ("Iyin", "Ado Ekiti"),
    ("Iyin", "Iworoko"),
    ("Ifaki", "Iworoko"),
    ("Ifaki", "Oye"),
    ("Iworoko", "Oye"),
    ("Iworoko", "Ado Ekiti"),
    ("Oye", "Ikole"),
    ("Ikole", "Omuo-Oke"),
    ("Ikole", "Ise"),
    ("Omuo-Oke", "Ise"),
    ("Ado Ekiti", "Ikere-Ekiti"),
    ("Ado Ekiti", "Ise"),
    ("Ikere-Ekiti", "Ise"),
]

# 2006 National Population Commission census, at Local Government Area (LGA)
# resolution -- Nigeria has no finer public town-level breakdown. Ado Ekiti,
# Ijero, Ikole, Ikere-Ekiti, Oye, Omuo-Oke, Efon Alaaye, Ise and Otun Ekiti
# are each their LGA's sole town in this network, so they take that LGA's
# full census figure directly. Three LGAs contain more than one of this
# network's towns (Ekiti West: Aramoko/Oke-Mesi/Ikogosi; Ido-Osi: Ifaki/Ido
# Ekiti; Irepodun/Ifelodun: Iworoko/Iyin); those LGA totals are split among
# their towns using each LGA's administrative headquarters (generally the
# largest settlement) to anchor the largest share -- an explicit estimate,
# since no census split below LGA level is publicly available.
population = [
    146_496,   # Otun Ekiti     (Moba LGA)
    72_000,    # Ido Ekiti      (Ido-Osi LGA, 45% split with Ifaki)
    165_099,   # Ijero          (Ijero LGA)
    53_968,    # Oke-Mesi       (Ekiti West LGA, 30% split)
    86_941,    # Efon Alaaye    (Efon LGA)
    35_978,    # Ikogosi        (Ekiti West LGA, 20% split)
    89_946,    # Aramoko        (Ekiti West LGA, 50% split -- LGA HQ)
    58_117,    # Iyin           (Irepodun/Ifelodun LGA, 45% split)
    88_001,    # Ifaki          (Ido-Osi LGA, 55% split)
    71_032,    # Iworoko        (Irepodun/Ifelodun LGA, 55% split -- LGA HQ area)
    134_210,   # Oye            (Oye LGA)
    168_436,   # Ikole          (Ikole LGA)
    137_955,   # Omuo-Oke       (Ekiti East LGA)
    313_690,   # Ado Ekiti      (Ado Ekiti LGA)
    147_255,   # Ikere-Ekiti    (Ikere LGA)
    113_754,   # Ise            (Ise/Orun LGA)
]

idx = {name: i for i, name in enumerate(labels)}
n = len(labels)

A = np.zeros((n, n), dtype=int)
for a, b in edges:
    i, j = idx[a], idx[b]
    A[i, j] = A[j, i] = 1

if __name__ == "__main__":
    # Plain matrix
    print(A)

    # Labelled view (needs pandas)
    try:
        import pandas as pd
        df = pd.DataFrame(A, index=labels, columns=labels)
        print(df.to_string())
        print("\nDegree per town:")
        print(df.sum(axis=1).sort_values(ascending=False).to_string())
    except ImportError:
        pass
