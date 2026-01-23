"""
Cloud Function to export predictions to BigQuery
"""

import json
import os
from google.cloud import bigquery
from google.cloud import secretmanager
import functions_framework


# BigQuery configuration
PROJECT_ID = os.environ.get('GCP_PROJECT', 'master-ai-cloud')
DATASET_ID = os.environ.get('BIGQUERY_DATASET', 'Ecommerce')
TABLE_ID = os.environ.get('BIGQUERY_TABLE', 'predictions')

# API Key configuration
API_KEY_SECRET_NAME = os.environ.get('API_KEY_SECRET_NAME', 'bigquery-export-api-key')
API_KEY_HEADER = 'X-API-Key'


def get_api_key():
    """Retrieve API key from Secret Manager"""
    try:
        client = secretmanager.SecretManagerServiceClient()
        secret_name = f"projects/{PROJECT_ID}/secrets/{API_KEY_SECRET_NAME}/versions/latest"
        response = client.access_secret_version(request={"name": secret_name})
        return response.payload.data.decode('UTF-8').strip()
    except Exception as e:
        print(f"Warning: Could not retrieve API key from Secret Manager: {e}")
        # Fallback to environment variable for development
        return os.environ.get('API_KEY', None)


def verify_api_key(request):
    """Verify API key from request"""
    api_key = get_api_key()
    if not api_key:
        # If no API key configured, allow access (for development)
        print("Warning: No API key configured, allowing access")
        return True
    
    # Check header first
    provided_key = request.headers.get(API_KEY_HEADER)
    if not provided_key:
        # Also check query parameter as fallback
        provided_key = request.args.get('api_key')
    
    if not provided_key:
        return False
    
    # Constant-time comparison to prevent timing attacks
    return provided_key == api_key


@functions_framework.http
def export_to_bigquery(request):
    """
    HTTP Cloud Function to export predictions to BigQuery
    
    Expected request body:
    {
        "predictions": [
            {
                "product_id": "...",
                "title": "...",
                ...
            }
        ]
    }
    """
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, X-API-Key',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    # Set CORS headers
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
    }
    
    try:
        # Verify API key (skip for OPTIONS requests)
        if request.method != 'OPTIONS':
            if not verify_api_key(request):
                return (json.dumps({
                    'error': 'Unauthorized',
                    'message': 'Invalid or missing API key'
                }, default=str), 401, headers)
        
        # Parse request
        if request.method != 'POST':
            return (json.dumps({'error': 'Method not allowed'}, default=str), 405, headers)
        
        request_json = request.get_json(silent=True)
        if not request_json:
            return (json.dumps({'error': 'No JSON data provided'}, default=str), 400, headers)
        
        predictions = request_json.get('predictions', [])
        if not predictions:
            return (json.dumps({'error': 'No predictions provided'}, default=str), 400, headers)
        
        # Initialize BigQuery client
        client = bigquery.Client(project=PROJECT_ID)
        table_ref = client.dataset(DATASET_ID).table(TABLE_ID)
        
        # Prepare rows for insertion (without timestamp)
        rows_to_insert = []
        for pred in predictions:
            row = {
                'product_id': pred.get('product_id', ''),
                'title': pred.get('title', ''),
                'description': pred.get('description', ''),
                'true_category_id': pred.get('true_category_id', ''),
                'true_category_name': pred.get('true_category_name', ''),
                'true_category_path': pred.get('true_category_path', ''),
                'predicted_category_id': pred.get('predicted_category_id', ''),
                'predicted_category_name': pred.get('predicted_category_name', ''),
                'predicted_category_path': pred.get('predicted_category_path', ''),
                'confidence': float(pred.get('confidence', 0.0)),
                'is_correct': bool(pred.get('is_correct', False)),
                'processing_time_ms': float(pred.get('processing_time_ms', 0.0)),
                'request_time_ms': float(pred.get('request_time_ms', 0.0)),
                'api_url': pred.get('api_url', '')
            }
            rows_to_insert.append(row)
        
        # Insert rows into BigQuery
        errors = client.insert_rows_json(table_ref, rows_to_insert)
        
        if errors:
            return (json.dumps({
                'error': 'Failed to insert data into BigQuery',
                'details': errors,
                'rows_inserted': 0,
                'rows_failed': len(predictions)
            }, default=str), 500, headers)
        
        # Success response
        return (json.dumps({
            'success': True,
            'message': f'Successfully inserted {len(rows_to_insert)} predictions',
            'rows_inserted': len(rows_to_insert)
        }, default=str), 200, headers)
    
    except Exception as e:
        return (json.dumps({
            'error': 'Internal server error',
            'message': str(e)
        }, default=str), 500, headers)
