# Skyjo RL

Une IA pour le jeu de cartes **Skyjo**, entraînée par apprentissage par renforcement
(MaskablePPO de [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)).
Un seul réseau joue à 2, 3 et 4 joueurs. Le dépôt contient le moteur de jeu, l'environnement
Gymnasium, des bots écrits à la main, l'entraînement, les scripts d'évaluation, une partie en
console, une petite interface web locale pour jouer contre l'IA, et une démo qui tourne
entièrement dans le navigateur.

L'IA est **battable** : elle bat nettement des bots simples à 2 joueurs, mais son avance fond à
4 joueurs, et Skyjo laisse une grande place au hasard des cartes.

Projet personnel et non officiel, sans lien avec l'éditeur du jeu Skyjo.

## Jouer dans le navigateur

**Démo : <https://Alban-dot.github.io/play-skyjo-ai/>** (2 à 4 joueurs, sur ordinateur ou
téléphone, rien à installer).

![Une partie à 4 joueurs dans la démo : les plateaux des trois IA en haut, la pioche, la
défausse et les scores au centre, le plateau du joueur en bas, le déroulement à droite](docs/demo.png)

Le moteur de jeu et le modèle de référence tournent sur ton appareil grâce à
[Pyodide](https://pyodide.org) (Python et NumPy compilés en WebAssembly) : rien n'est envoyé à un
serveur, et le bilan et les parties restent dans le navigateur. Les fichiers de Pyodide 0.29.5 ne
sont pas dans ce dépôt : le workflow de publication (`.github/workflows/pages.yml`) les
télécharge depuis la distribution officielle de Pyodide (`cdn.jsdelivr.net/pyodide/v0.29.5/full/`)
avec `scripts/fetch_pyodide.py`, qui vérifie l'empreinte SHA-256 de chaque fichier, puis les sert
depuis le même site que la démo. Le premier chargement télécharge de 9 à 15 Mo selon la
compression du serveur.

## Le projet en bref

- **Moteur** (`skyjo/game/`) : règles complètes, indépendantes de l'apprentissage
  (voir [docs/RULES.md](docs/RULES.md)). Chaque joueur ne voit que l'information publique.
- **Environnement** (`skyjo/env/`) : observation de taille fixe (jusqu'à 8 joueurs grâce au
  remplissage), 26 actions avec masque des actions légales, récompense = écart de score.
- **Bots** (`skyjo/agents/`) : aléatoire, heuristique, « prudent » (heuristique qui évite de
  finir la manche quand il risque le doublement), chasseur de colonnes.
- **Entraînement** (`skyjo/training/`) : le réseau a d'abord copié le bot prudent (clonage de
  comportement), puis a été entraîné par PPO à 2 joueurs, puis à 2–4 joueurs contre un mélange
  de bots, de ses anciennes versions et de modèles figés.
- **Modèle de référence** : `models/frozen/ppo_multi_1200k_seed4.zip` (5,3 Mo), joué avec un
  garde-fou qui casse les boucles où deux joueurs se renverraient indéfiniment la même carte.

## Résultats

Méthode : les deux joueurs comparés reçoivent **les mêmes donnes** (parties appariées), les
places tournent d'une partie à l'autre, intervalles de confiance à 95 %. L'**écart** est le
score total du joueur moins la moyenne de ses adversaires : **négatif = le joueur marque moins
de points que ses adversaires = meilleur** (au Skyjo, le score le plus bas gagne). Une
**différence d'écart** se lit comme « variante − référence » sur les mêmes donnes : négative,
la variante fait mieux ; positive, elle fait moins bien.

### Le modèle de référence contre des bots

Contre des bots prudents (2 000 parties par table, graine 999012, avec le garde-fou) :

| Table | Victoires (hasard) | Écart |
|---|---|---|
| 2 joueurs | 67,0 % [65,0 ; 69,1] (50 %) | −16,3 [−18,0 ; −14,7] |
| 3 joueurs | 44,0 % [41,8 ; 46,2] (33 %) | −7,8 [−9,1 ; −6,4] |
| 4 joueurs | 28,9 % [27,0 ; 31,0] (25 %) | −1,7 [−3,0 ; −0,5] |

Contre des bots heuristiques (1 000 parties par table, graine 999002, sans le garde-fou) :

| Table | Victoires (hasard) | Écart |
|---|---|---|
| 2 joueurs | 80,2 % [77,6 ; 82,6] (50 %) | −33,4 [−35,9 ; −31,0] |
| 3 joueurs | 53,0 % [49,9 ; 56,1] (33 %) | −19,6 [−21,4 ; −17,8] |
| 4 joueurs | 38,0 % [35,0 ; 41,0] (25 %) | −13,9 [−15,5 ; −12,3] |

À 4 joueurs, le modèle n'a qu'une petite avance sur le bot prudent. Sur une autre graine
(999021, 1 000 parties), un bot prudent face à trois copies du modèle gagne 21,9 % des parties
pour un hasard de 25 %, avec un écart de +0,6 point : ils sont presque au même niveau.

### Ce qui a marché

**Compléter systématiquement les colonnes rentables.** Une règle ajoutée au modèle, au moment
du jeu, l'oblige à compléter une colonne dès que le gain théorique est positif. Différence
appariée d'écart, modèle avec règle − modèle seul (2 000 parties par table contre des prudents,
graine 999012) :

| Table | Différence d'écart | Différence de victoires |
|---|---|---|
| 2 joueurs | −1,9 [−3,4 ; −0,4] | +1,1 % [−1,0 ; +3,2] |
| 3 joueurs | −2,8 [−4,0 ; −1,5] | +3,0 % [+0,7 ; +5,3] |
| 4 joueurs | −3,5 [−4,6 ; −2,4] | +4,1 % [+1,9 ; +6,3] |

Le gain d'environ 3 points à 3 et 4 joueurs a été confirmé sur une seconde graine ; à 2
joueurs, l'effet reste incertain d'une graine à l'autre. C'est le mode « guard-forced » (le style « L'Expert » de
l'interface à 3 et 4 joueurs).

### Ce qui n'a rien apporté de démontré

- **Entraîner avec cette règle** plutôt que l'ajouter au jeu après coup : différence d'écart
  de +0,3 [−1,9 ; +2,4] à 2 joueurs, +0,9 [−0,7 ; +2,5] à 3, +0,2 [−1,3 ; +1,7] à 4. Rien de
  détectable.
- **Prolonger l'entraînement** d'un million de pas avec un taux d'apprentissage décroissant :
  différence d'écart avec le modèle de départ de +1,6 [−0,5 ; +3,7] à 2 joueurs (en sa
  défaveur, non significatif), −2,1 [−3,7 ; −0,4] à 3 et −1,8 [−3,3 ; −0,4] à 4. Une seule
  graine, non confirmé : au mieux environ 2 points.
- **Un bot qui parie sur les colonnes** en début de manche (chasseur de colonnes, réglage
  14/8, l'un des deux retenus d'une grille de 12, confirmé sur une graine neuve) : face aux bots
  prudents, différence d'écart avec le prudent de
  −1,6 [−4,5 ; +1,4], −1,0 [−3,3 ; +1,3] et −1,7 [−3,7 ; +0,2] à 2, 3 et 4 joueurs. Rien de
  détectable.
- **Forcer la fin de manche quand l'avance estimée dépasse un seuil** (5, 10, 15, 20 ; le
  modèle 1,2M décide seul sous le seuil, face aux bots prudents, mêmes donnes) : aucun gain
  démontré ; le seuil 5 dégrade nettement à 2 joueurs : il gagne 7,5 points de pourcentage de
  parties en moins (60,9 % contre 68,4 %, différence −7,5 [−10,3 ; −4,7]) et son écart de
  score est plus élevé de 7,2 points (différence d'écart +7,2 [+5,3 ; +9,1]) ; les deux vont
  dans le sens d'une dégradation. Aux seuils 15 et 20, les différences d'écart vont de −0,4 à
  +0,0 et aucun de leurs intervalles n'exclut 0.

### Contre des humains

Quelques parties seulement ont été jouées contre lui, et l'humain les a toutes gagnées. C'est
trop peu pour mesurer un niveau, mais cela suffit à dire que l'IA n'est pas imbattable.

## Installation

Python 3.11 ou plus récent (développé avec 3.13).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Commandes principales

```bash
# tests (les tests longs sont exclus par défaut ; pytest -m slow pour eux)
pytest

# jouer en console contre le modèle, à 2, 3 ou 4 joueurs
python scripts/play_vs_ai.py --players 3 --mode guard
python scripts/play_vs_ai.py --bilan             # bilan cumulé

# interface web locale : http://127.0.0.1:8000 (règles et aide sur /aide)
python scripts/run_ui.py
ADMIN_TOKEN=un-jeton-long python scripts/run_ui.py   # active /admin (le jeton sert de mot de passe)

# démo dans le navigateur, servie en local : http://127.0.0.1:8001
python scripts/fetch_pyodide.py      # une fois : Pyodide dans web_demo/pyodide/ (empreintes vérifiées)
python scripts/build_web_demo.py     # poids, modules et styles de la démo
python -m http.server 8001 --bind 127.0.0.1 --directory web_demo

# évaluer un modèle contre des bots, puis comparer deux modèles sur les mêmes donnes
python scripts/evaluate.py --model models/frozen/ppo_multi_1200k_seed4.zip --loop-guard \
    --opponents cautious heuristic --n-players 3 --n-games 1000 --seed 1 --per-game-csv a.csv
python scripts/compare_paired.py a.csv b.csv --names A B

# entraîner (long) à partir d'une configuration
python -m skyjo.training.train --config configs/two_players.yaml
```

L'interface web garde les parties terminées dans une base SQLite locale (`SKYJO_DB`, par défaut
`logs/skyjo.db`) et permet de revoir une partie pas à pas. Les styles d'IA proposés (du bot
aléatoire au modèle de référence) se règlent dans `skyjo/ui/styles.yaml`.

## Limites

- Les mesures opposent le modèle à des **bots**, pas à des humains.
- Le modèle de référence vient d'**un seul entraînement** (une graine) ; une autre graine
  donnerait un modèle un peu différent.
- Seul le modèle de référence est fourni. Les configurations qui partent d'un modèle
  intermédiaire (`init_from`) supposent de refaire la chaîne : démonstrations du bot prudent
  (`scripts/collect_demos.py`), copie (`scripts/train_copy.py`), puis entraînements successifs.
- Le réseau accepte jusqu'à 8 joueurs, mais il n'a été entraîné et mesuré qu'à 2, 3 et 4.
- Une petite variante de règle est tranchée par le moteur (ordre de défausse lors d'une
  colonne complète, voir [docs/RULES.md](docs/RULES.md)).
- L'interface web est en français et prévue pour un usage local ; elle n'a pas été éprouvée
  avec beaucoup de joueurs simultanés.

## À propos

Développé avec l'aide de Claude Code ; j'ai défini les expériences, relu et validé les résultats.

## Licence

MIT, voir [LICENSE](LICENSE).
