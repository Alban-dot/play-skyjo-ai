# Règles de Skyjo utilisées par le moteur

Le moteur (`skyjo/game/`) applique les règles ci-dessous ; certains points, que la règle
officielle laisse ouverts, sont tranchés ici et signalés comme tels.

## Cartes

150 cartes : −2 (5 cartes), −1 (10), 0 (15), puis 1 à 12 (10 cartes chacune).

## Mise en place d'une manche

- Chaque joueur reçoit 12 cartes face cachée, en grille de 3 lignes × 4 colonnes.
- Une carte de la pioche est retournée pour démarrer la défausse.
- Chaque joueur retourne 2 de ses cartes. **À chaque manche**, le joueur dont ces deux cartes
  font la plus grande somme commence ; en cas d'égalité, celui qui a le plus de points au total
  de la partie ; s'il reste une égalité, tirage au sort.

## Un tour

Le joueur choisit :

- **A. Piocher** une carte face cachée, puis
  - soit la **poser** à un emplacement de sa grille (carte visible ou cachée) : la carte
    remplacée est défaussée, face visible ;
  - soit la **défausser** et **retourner** une de ses cartes cachées.
- **B. Prendre la carte du dessus de la défausse** et la **poser obligatoirement** dans sa
  grille ; la carte remplacée est défaussée.

## Colonnes

Dès qu'une colonne contient 3 cartes **visibles et de même valeur**, elle est retirée de la
grille et part à la défausse ; elle ne compte plus. Ordre retenu : la carte remplacée est
défaussée d'abord, puis les 3 cartes de la colonne (le dessus de la défausse est donc une carte
de la colonne). Une colonne complète est retirée à tout moment, y compris lors du retournement
final des cartes en fin de manche.

## Fin de manche

- Dès qu'un joueur a **toutes ses cartes visibles**, chacun des autres joue **un dernier tour**.
- Toutes les cartes restantes sont retournées, **puis** les colonnes de 3 cartes identiques
  ainsi formées sont retirées, **puis** on compte les points (somme des cartes).
- **Doublement** : si le joueur qui a terminé la manche n'a pas **strictement** le plus petit
  score de la manche, son score de manche est doublé (s'il est positif). Une égalité pour le
  plus petit score déclenche aussi le doublement.

## Fin de partie

La partie se termine quand un joueur atteint ou dépasse **100 points** au total. Le total le
plus bas gagne ; en cas d'égalité, tous les ex æquo gagnent. À 3 joueurs et plus, seule la
première place compte comme une victoire.

## Pioche vide

La défausse, sauf sa carte du dessus, est mélangée pour refaire une pioche.

## Hors des règles

Le moteur peut arrêter de force une manche qui ne se termine pas (deux joueurs qui se
renvoient indéfiniment la même carte) : toutes les cartes sont alors comptées, sans doublement.
Cela sert à l'entraînement et aux simulations, jamais dans une partie normale.
