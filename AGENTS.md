# AGENTS.md — Protocol d'exécution des agents IA

## Source de vérité et responsabilités
- ChatGPT pilote les missions : inspecte les éléments publiés sur GitHub, fixe l'objectif, le périmètre et les critères d'acceptation, puis contrôle le résultat.
- OpenCode exécute dans l'environnement local : lit le dépôt, modifie le code, utilise les outils réellement disponibles, lance les tests et publie les changements.
- GitHub est la source de vérité commune pour les commits poussés, branches, PR, rapports et résultats CI.
- L'utilisateur ne fait que transmettre le prompt ChatGPT à OpenCode puis le rapport OpenCode à ChatGPT. Ne lui délègue pas l'analyse du diff ou les validations techniques ordinaires.

## Cycle de mission obligatoire
1. Vérifier le dépôt, le remote, la branche, le SHA courant et `git status --short` avant toute modification.
2. Lire les instructions locales et les fichiers d'état pertinents. Ne pas refaire un audit général lorsqu'un diagnostic ciblé suffit.
3. Définir une tâche cohérente, un périmètre limité et un critère de réussite mesurable.
4. Préserver tous les changements locaux préexistants. Ne jamais les écraser, les nettoyer ou les réinitialiser.
5. Travailler sur une branche dédiée pour les changements applicatifs, sauf instruction explicite différente.
6. Effectuer les tests réellement pertinents, inspecter le diff complet et lancer `git diff --check`.
7. Si les vérifications passent, créer un commit ciblé et pousser la branche. Ne pas pousser directement sur la branche par défaut pour les changements applicatifs ordinaires.
8. Ne pas ouvrir une PR fictive : rapporter l'URL seulement après confirmation de GitHub.

## Rapport obligatoire
Terminer chaque mission avec ce format compact :

```text
MISSION: <nom court>
REPOSITORY: <owner/repo>
BRANCH: <branche>
COMMIT: <SHA complet ou aucun>
PUSH: oui/non
MODIFICATIONS: <résumé et fichiers essentiels>
TESTS: <commandes exactes et résultats réels>
CI: <URL et statut, ou non déclenchée>
TESTS LOCAUX RÉELS: <faits et résultats>
NON VÉRIFIÉ: <limites restantes, ou aucun>
BLOCAGE: oui/non — <raison>
```

## Boucle de contrôle
À chaque rapport, ChatGPT doit vérifier directement sur GitHub le commit, le diff réel et les résultats CI accessibles. Le rapport seul n'est jamais une preuve suffisante. ChatGPT renvoie ensuite un prompt correctif précis, le prompt de la mission suivante ou un bref statut TERMINÉ. Ne pas demander de validation ordinaire à l'utilisateur.

## Efficacité
- Une tâche cohérente par cycle ; éviter audits répétitifs, refontes non justifiées et documentation sans utilité opérationnelle.
- Modification minimale, réutiliser les mécanismes existants et ajouter une dépendance seulement si nécessaire et justifiée.
- Après deux échecs de la même approche, changer de méthode ou signaler un blocage ; ne pas répéter le même essai.
- Séparer tests unitaires, intégration, navigateur/runtime réel et production. Ne jamais transformer une simulation en preuve réelle.
- Ne jamais inventer des résultats, des métriques, des tests ou des fonctionnalités.

## Sécurité et opérations
- Utiliser uniquement les outils et accès réellement disponibles. Si un outil manque, le signaler clairement.
- Ne jamais publier de secrets, tokens, données personnelles ou fichiers locaux sensibles.
- Jamais de force-push, réécriture d'historique ou suppression de données.
- Les règles locales spécifiques au produit, aux paiements, aux données, à la sécurité et à la production priment sur ce protocole commun.
- Pas de migration destructive, fusion sur la branche par défaut ou déploiement de production sans autorisation explicite de l'utilisateur.
- Un commit de documentation ou de consignes ne doit pas provoquer un déploiement applicatif inutile ; configurer les filtres de chemins dans les workflows concernés plutôt que désactiver les vérifications utiles.
