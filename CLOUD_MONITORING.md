# Guide Cloud Monitoring et Observabilité

Ce guide explique comment utiliser Cloud Monitoring et Cloud Trace pour observer l'API de classification.

## Prérequis

1. **Projet GCP configuré** avec les APIs suivantes activées:
   - Cloud Run API
   - Cloud Monitoring API
   - Cloud Trace API
   - Cloud Build API

2. **Authentification** :
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Permissions** : Le service Cloud Run doit avoir les permissions:
   - `roles/cloudtrace.agent` (pour Cloud Trace)
   - `roles/monitoring.metricWriter` (pour Cloud Monitoring)

## Déploiement

1. **Déployer l'API sur Cloud Run** :
   ```bash
   chmod +x deploy_cloud_run.sh
   export GOOGLE_CLOUD_PROJECT=your-project-id
   ./deploy_cloud_run.sh
   ```

2. **Vérifier que l'instrumentation fonctionne** :
   - L'API détecte automatiquement qu'elle tourne sur GCP via la variable `GOOGLE_CLOUD_PROJECT`
   - Les métriques sont exportées toutes les 60 secondes
   - Les traces sont envoyées en batch

## Dashboard Cloud Monitoring

### Créer un dashboard personnalisé

1. **Accéder à Cloud Monitoring** :
   - Console GCP → Monitoring → Dashboards
   - Cliquer sur "Create Dashboard"

2. **Métriques disponibles** :

   #### Latence (Request Duration)
   - **Métrique** : `custom.googleapis.com/opentelemetry/api_request_duration_seconds`
   - **Graphique recommandé** : Line chart avec percentiles (p50, p95, p99)
   - **Filtres** : `endpoint="/classify"`

   #### Throughput (Request Rate)
   - **Métrique** : `custom.googleapis.com/opentelemetry/api_requests_total`
   - **Graphique recommandé** : Line chart (rate par seconde)
   - **Filtres** : `endpoint="/classify"`, `status="2xx"`

   #### Taux d'erreur
   - **Métrique** : `custom.googleapis.com/opentelemetry/api_errors_total`
   - **Graphique recommandé** : Line chart
   - **Filtres** : `error_type="5xx"` ou `error_type="4xx"`

   #### Score de confiance
   - **Métrique** : `custom.googleapis.com/opentelemetry/api_confidence_score_average`
   - **Graphique recommandé** : Line chart
   - **Seuil d'alerte** : < 0.7

   #### Temps d'inférence
   - **Métrique** : `custom.googleapis.com/opentelemetry/api_inference_duration_seconds`
   - **Graphique recommandé** : Line chart avec percentiles

### Dashboard recommandé (5 widgets)

1. **Latence p95** (ligne rouge à 500ms)
2. **Throughput** (requêtes/seconde)
3. **Taux d'erreur** (ligne rouge à 1%)
4. **Score de confiance moyen**
5. **Temps d'inférence p95**

## Cloud Trace

### Visualiser les traces

1. **Accéder à Cloud Trace** :
   - Console GCP → Trace → Trace List

2. **Filtrer les traces** :
   - Service : `ecommerce-classification-api`
   - Endpoint : `/classify`

3. **Spans disponibles** :
   - `POST /classify` : Span principal de la requête
   - `model.predict` : Span pour la prédiction complète
   - `model.embedding` : Span pour la génération d'embedding
   - `model.classify` : Span pour la classification

4. **Analyser les performances** :
   - Identifier les requêtes lentes (> 1s)
   - Voir quelle étape prend le plus de temps (embedding vs classification)
   - Analyser les erreurs avec les stack traces

## Alertes

### Créer des alertes

1. **Accéder aux Alertes** :
   - Console GCP → Monitoring → Alerting → Create Policy

2. **Alertes recommandées** :

   #### Latence élevée
   - **Condition** : `api_request_duration_seconds` p95 > 500ms
   - **Durée** : 5 minutes
   - **Notification** : Email ou Slack

   #### Taux d'erreur élevé
   - **Condition** : `api_errors_total` rate > 1% des requêtes
   - **Durée** : 5 minutes

   #### Score de confiance faible
   - **Condition** : `api_confidence_score_average` < 0.7
   - **Durée** : 10 minutes

   #### Modèle non disponible
   - **Condition** : Health check `/health` retourne `unhealthy`
   - **Durée** : 1 minute

## Métriques Prometheus (local)

Pour le développement local, les métriques Prometheus restent disponibles sur `/metrics` :
- Compatible avec Grafana
- Format standard Prometheus

## Troubleshooting

### Les métriques n'apparaissent pas

1. Vérifier que `GOOGLE_CLOUD_PROJECT` est défini dans Cloud Run
2. Vérifier les permissions IAM du service
3. Vérifier les logs Cloud Run pour les erreurs d'export

### Les traces n'apparaissent pas

1. Vérifier que Cloud Trace API est activée
2. Vérifier les permissions `roles/cloudtrace.agent`
3. Attendre quelques minutes (traces envoyées en batch)

### Performance

- Les métriques sont exportées toutes les 60 secondes (configurable)
- Les traces sont envoyées en batch (latence ~10-30 secondes)
- Impact minimal sur les performances de l'API

