import matplotlib.pyplot as plt

def rpt_multi(models, t, s, d, npoint, filetype="pdf", title="Comparison of models"):
    """
    Superpose plusieurs modèles sur le même graphe et compile leurs textes sous le graphique.
    
    Parameters
    ----------
    models : list of tuples
        Chaque tuple: (rpt_func, params, stats, name, optional_color)
    t, s : array_like
        Données observées
    d : tuple
        (q, r) pumping rate et distance
    npoint : int
        Nombre de points pour dérivée
    filetype : str
        "pdf" ou "png"
    title : str
        Titre du graphique
    """
  
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Drawdown (m)")
    ax.grid(True, which="both", ls="--", lw=0.5)
    # Liste pour stocker tous les textes des modèles
    all_texts = []
    
    colors = ["#6A0DAD", "#00CED1", "#FF1493", "#228B22", "#FF8C00", "#1E90FF"]
    for i, mod in enumerate(models, start=1):
        if len(mod) == 5:
            func, p, stats, name, color = mod
        else:
            func, p, stats, name = mod
            color = colors[i % len(colors)]

        # Appel de la fonction rpt sur le même axe
        fig_mod, ax_mod = func(
            p, stats, t, s, d, npoint, ax=ax, color=color)
        lines = ax_mod.get_lines()
        # lignes modèles uniquement
        lines[1].set_color(color)      # modèle
        lines[3].set_color(color)      # modèle dérivée
        # Formatage du texte du modèle
        model_texts = [txt.get_text() for txt in fig_mod.texts]

        # On formate chaque bloc
        model_block = f"Model {i}: {name}\n" + "\n".join(model_texts)
        all_texts.append(model_block)

        # Optionnel : nettoyer fig_mod.texts si nécessaire
        fig_mod.texts.clear()

    # Concaténer tous les blocs
    combined_text = "\n\n".join(all_texts)

    fig.text(
        1.02, 0.5, combined_text, ha='left', va='center',
        fontsize=10, family='arial',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    handles, labels = ax.get_legend_handles_labels()
    from collections import OrderedDict
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    fig.tight_layout()

    if filetype == "pdf":
        fig.savefig("comparison_report.pdf", bbox_inches="tight")
    elif filetype == "png":
        fig.savefig("comparison_report.png", bbox_inches="tight")

    return fig, ax
