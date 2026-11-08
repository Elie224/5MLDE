# Données : provenance, conditions d'utilisation, simulation

## Source

| | |
|---|---|
| Jeu de données | **Online Retail**, UCI Machine Learning Repository |
| Page | https://archive.ics.uci.edu/dataset/352/online+retail |
| Archive téléchargée | https://archive.ics.uci.edu/static/public/352/online+retail.zip |
| DOI | 10.24432/C5BW33 |
| Licence | **Creative Commons Attribution 4.0 (CC BY 4.0)** : réutilisation et modification autorisées, y compris commerciales, sous réserve de citer la source |
| Producteur | Daqing Chen, School of Engineering, London South Bank University |
| Citation | Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33 |
| Référence associée | Chen, D., Sain, S. L., Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing and Customer Strategy Management*, 19(3), 197–208. |

Le dataset décrit toutes les transactions d'un e-commerce britannique (articles cadeaux,
clientèle en partie grossiste) entre le **01/12/2010 et le 09/12/2011** : environ 541 900
lignes de facture. Plusieurs pays sont présents ; le **périmètre métier** (Royaume-Uni,
top 50 produits) est appliqué au nettoyage (phase 2), pas à l'ingestion.

**Données personnelles** : la seule information client est `CustomerID`, un identifiant
numérique pseudonymisé à la source. Il n'est **pas utilisé** par le modèle (prévision
agrégée par produit) et n'est jamais exposé par l'API.

**Versionnement** : les données ne sont pas versionnées dans Git (taille, licence de
redistribution). Elles sont récupérées par `retail data` ; le SHA256 de l'archive est figé
dans `configs/data.yaml` et chaque lot porte une empreinte de contenu dans
`data/batches/manifest.json`.

## Schéma source

| Colonne | Type (après ingestion) | Description |
|---|---|---|
| InvoiceNo | str | n° de facture ; préfixe `C` = annulation |
| StockCode | str (majuscules) | code produit ; quelques codes techniques (`POST`, `M`, `D`, `DOT`, `BANK CHARGES`…) |
| Description | str | libellé produit |
| Quantity | int64 | quantité ; négative pour les retours/annulations |
| InvoiceDate | datetime | date et heure de facturation |
| UnitPrice | float64 | prix unitaire en livres sterling |
| CustomerID | Int64 (nullable) | identifiant client, ~25 % manquant |
| Country | str | pays du client |

L'ingestion (phase 1) **ne filtre rien** : elle type les colonnes et trace la version. Le
nettoyage métier et les contrôles bloquants relèvent de la phase 2, pour que chaque ligne
écartée soit comptée et justifiée dans un rapport JSON (`data/reports/quality/`).

Ordre bloquant d'un lot : **InvoiceSchema** (colonnes, types, dates dans la semaine ISO,
volume minimum, part de prix ≤ 0) → nettoyage (hors Royaume-Uni, annulations `C`,
quantités ≤ 0, prix ≤ 0, codes non-produits, valeurs aberrantes) → agrégation
1 ligne / SKU / semaine → **PanelSchema** (quantités ≥ 0, pas de doublon SKU×semaine).
Un lot hors schéma n'écrit pas de panel et n'enchaîne pas sur l'entraînement.

## Simulation de l'arrivée des données

Le dataset est **historique**. Pour démontrer une chaîne opérationnelle, il est **rejoué
par lots hebdomadaires** :

- un lot = les factures d'une **semaine ISO** (lundi 00:00 → dimanche 23:59),
  fichier `data/batches/<AAAA-Www>.parquet` ;
- la première semaine (début mercredi 01/12/2010) et la dernière (fin vendredi 09/12/2011)
  sont **incomplètes** et écartées (`batching.drop_partial_weeks`) ;
- il reste **51 lots** de `2010-W49` à `2011-W48` : **2010-W52 est absente** (aucune
  facture, fermeture de Noël). Ce trou est listé dans `data/batches/manifest.json`
  (`missing_weeks`), ce n'est pas une série continue de 52 semaines ;
- l'orchestrateur (phase 5) « livre » un lot à chaque déclenchement, comme si la semaine
  venait de se terminer ;
- l'arrivée du lot N+1 fournit les ventes réelles de la semaine prévue en N : c'est la
  source du **feedback automatique** (phase 7).

**Démarrage à froid** : les variables de retard vont jusqu'à −4 semaines avec une moyenne
mobile sur 4 semaines. Les `warmup_weeks` (8) premières semaines servent uniquement
d'historique, et la liste des produits suivis (top 50) est calculée **sur ces seules
semaines** puis figée, pour éviter toute fuite d'information future.

**`--until-week` (rejeu)** : l'entraînement n'utilise que les lots ISO ≤ cette semaine.
Il faut au minimum les 8 semaines de warmup, **plus une semaine d'entraînement**, **plus
`test_weeks` (8) semaines de hold-out**. Avec le calendrier UCI (premier lot `2010-W49`),
la première valeur possible est **`2011-W14`**. En dessous : `TrainError`.

La promotion compare toujours les modèles **sur les semaines de test du challenger** (rejeu
`--until-week` inclus) : on ne compare pas des MAE calculées sur des fenêtres différentes.

**Rejeu opérationnel** (`retail replay`) : un seul lot par tick, contrôlé avec
`retail quality --week AAAA-Www` (pas de revalidation des 51 lots). Un lot hors
schéma **bloque le curseur sur cette semaine** jusqu'à correction (pas de saut au
lot suivant). Historique : `data/reports/orchestration/runs.jsonl` et table
`pipeline_runs`. Alertes : toujours POST vers `RETAIL_ALERT_WEBHOOK_URL`
(`http://127.0.0.1:8001/alerts`, fichier `alerts_received.jsonl` + table `alerts`).
Copie locale `alerts.jsonl` seulement si le webhook échoue.

Quand le lot d'une semaine est validé, les quantités du panel ferment les prédictions
API de **cette** semaine (`feedbacks.origin = auto`) : c'est la WAPE en production
par `model_version`.
