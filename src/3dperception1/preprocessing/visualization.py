"""Visualizzazioni dei point cloud LiDAR.

Tre plot sullo stesso frame, per capire cosa fa ogni passo della pipeline.
Ogni funzione disegna su una figura propria e la chiude: il grafico non
accumulerebbe tracce da una chiamata all'altra.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# il file vive in src/3dperception1/preprocessing/, quindi i moduli fratelli
# stanno nella stessa cartella
sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_velodyne import load_velodyne  # noqa: E402
from road_removal import fit_ground  # noqa: E402

FRAME = 100  # frame di prova: scena urbana con auto, pedone, ciclista

# parents[0]=preprocessing  [1]=3dperception1  [2]=src  [3]=radice del repo
OUT = Path(__file__).resolve().parents[3] / "outputs" / "figures"


def plot_nuvola_grezza(frame: int = FRAME) -> None:
    """Step 0 - la nuvola come arriva dal sensore, senza filtri.

    Due viste. In alto: x contro y, cioe' dall'alto. In basso: x contro z,
    cioe' di lato. Colori diversi perche' i due plot hanno domande diverse.
    """
    punti = load_velodyne(frame)
    x, y, z, intensita = punti.T  # la trasposta mette ogni colonna in una variabile

    fig, ax = plt.subplots(2, 1, figsize=(18, 12))

    # vista dall'alto: z indica l'altezza, quindi le auto sono tinte diverse dal fondo
    ax[0].scatter(x, y, s=0.05, c=z, cmap="viridis", rasterized=True)
    ax[0].set_title(f"frame {frame} - vista dall'alto, colore = z")
    ax[0].set_ylabel("y [m] sinistra")

    # vista laterale: y indica il lato, serve a controllare il piano terra
    ax[1].scatter(x, z, s=0.05, c=y, cmap="viridis", rasterized=True)
    ax[1].set_title("vista laterale, colore = y")
    ax[1].set_ylabel("z [m]")

    for a in ax:
        a.set_aspect("equal")
        a.set_xlabel("x [m] avanti")
    plt.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT / f"step0_nuvola_grezza_{frame}.png", dpi=110)
    plt.close(fig)  # libera la memoria: 123.000 punti per figura occupano parecchio
    plt.show()


def plot_zoom_densita(frame: int = FRAME) -> None:
    """Confronto colore vs bianco e nero sulla stessa vista.

    Il colore serve a distinguere le altezze. Il nero serve a contare la
    densita': con molti punti sovrapposti la densita' e' piu' leggibile in
    scala di grigi che con una colormap.
    Il filtro z > -1.2 e' una soglia grella, non il piano vero: serve solo
    a togliere l'asfalto per guardare sotto. Dopo lo step 2 non serve piu'.
    """
    punti = load_velodyne(frame)
    x, y, z = punti[:, 0], punti[:, 1], punti[:, 2]

    sopra = z > -1.2  # maschera booleana: True dove il criterio e' soddisfatto

    fig, ax = plt.subplots(1, 2, figsize=(20, 9))

    ax[0].scatter(x[sopra], y[sopra], s=1.5, c=z[sopra], cmap="turbo", rasterized=True)
    ax[0].set_title("vista dall'alto, solo z > -1.2, colore = z")

    ax[1].scatter(x[sopra], y[sopra], s=1.5, c="black", rasterized=True)
    ax[1].set_title("stessa vista in bianco e nero: meglio la densita'")

    for a in ax:
        a.set_aspect("equal")
        a.set_xlim(-5, 25)  # la scena utile sta qui, oltre e' rumore lontano
        a.set_ylim(-15, 15)
        a.set_xlabel("x [m] avanti")
        a.set_ylabel("y [m] sinistra")
    plt.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT / f"step0_zoom_densita_{frame}.png", dpi=110)
    plt.close(fig)
    plt.show()


def plot_ground_removal(frame: int = FRAME) -> None:
    """Step 2 - cosa ha tolto RANSAC.

    La riga tratteggiata e' il piano stimato. Se taglia il bordo superiore
    della striscia grigia, il piano e' leggermente sbagliato: e' il controllo
    piu' rapido che esista.
    """
    punti = load_velodyne(frame)[:, :3]  # solo xyz, l'intensita' non serve qui
    normale, offset, inlier = fit_ground(punti)

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    # vista laterale, per capire se il piano ha preso solo l'asfalto
    ax[0].scatter(punti[inlier, 0], punti[inlier, 2], s=0.4, c="lightgray",
                  label="asfalto (inlier)", rasterized=True)
    ax[0].scatter(punti[~inlier, 0], punti[~inlier, 2], s=0.4, c="darkorange",
                  label="sopra il piano", rasterized=True)
    ax[0].axhline(-offset, color="k", ls="--", lw=1, label="piano stimato")
    ax[0].set_title("vista laterale")
    ax[0].legend(loc="upper right")

    # vista dall'alto. Il colore e' l'altezza SOPRA IL PIANO, non z:
    # se la strada e' in salita z non coincide con l'altezza reale dal basso
    sopra = punti[~inlier]  # ~ inverte la maschera: prende i NON inlier
    altezza = sopra @ normale + offset
    ax[1].scatter(sopra[:, 0], sopra[:, 1], s=1.2, c=altezza, cmap="turbo",
                  rasterized=True)
    ax[1].set_title("vista dall'alto, colore = altezza sul piano")
    ax[1].set_xlim(-5, 25)
    ax[1].set_ylim(-15, 15)

    for a in ax:
        a.set_aspect("equal")
        a.set_xlabel("x [m] avanti")
    ax[0].set_ylabel("z [m]")
    ax[1].set_ylabel("y [m] sinistra")

    plt.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT / f"step2_ground_removal_{frame}.png", dpi=110)
    plt.close(fig)
    plt.show()

    # numeri di controllo: l'istogramma delle altezze dice piu' del grafico
    print(f"  tolto dal piano : {inlier.sum()} di {len(punti)}")
    print(f"  candidati       : {len(sopra)}")
    for lo, hi, nome in [
        (0.2, 0.5, "0.2-0.5 m  basso"),
        (0.5, 2.0, "0.5-2.0 m  pedoni e auto"),
        (2.0, 5.0, "2.0-5.0 m  furgoni"),
        (5.0, 99.0, "> 5 m     muri e pali"),
    ]:
        n = int(np.count_nonzero((altezza >= lo) & (altezza < hi)))
        print(f"    {nome:<26} {n:>7}")


plot_nuvola_grezza()
plot_zoom_densita()
plot_ground_removal()