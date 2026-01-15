import matplotlib.pyplot as plt

# Dimensions du tableau
n_cols = 10  # Nombre de colonnes (cases)

# Créer une figure
fig, ax = plt.subplots(figsize=(8, 2))

# Tracer les bordures des cases
for i in range(n_cols + 1):
    ax.plot([i, i], [0, 1], color="black", linewidth=1)  # Lignes verticales
ax.plot([0, n_cols], [0, 0], color="black", linewidth=1)  # Ligne horizontale inférieure
ax.plot([0, n_cols], [1, 1], color="black", linewidth=1)  # Ligne horizontale supérieure

# Personnaliser l'apparence
ax.set_xlim(0, n_cols)
ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.axis("off")  # Masquer les axes
ax.set_title("centres")
# Sauvegarder et afficher
plt.tight_layout()
plt.savefig("tableau_1D_vide.png", dpi=300)
plt.show()
