"""
Simple script to test the classification API
"""

import requests
import time

# Change this to your Cloud Run URL (or localhost for local testing)
API_URL = "https://ecommerce-classification-api-pjplfwgp5q-ew.a.run.app"
#API_URL = "http://localhost:8000"

# Sample products to test
TEST_PRODUCTS = [
    {"title": "Samsung Galaxy S21", "description": "Android smartphone with 6.2 inch screen"},
    {"title": "iPhone 14 Pro Max", "description": "Apple smartphone with A16 chip"},
    {"title": "MacBook Pro 16 M2", "description": "Apple laptop with M2 Pro chip"},
    {"title": "Sony WH-1000XM5", "description": "Wireless noise cancelling headphones"},
    {"title": "Nikon D850", "description": "Professional DSLR camera 45.7MP"},
    {"title": "Samsung 65 inch QLED TV", "description": "4K smart TV with HDR"},
    {"title": "Dyson V15 Detect", "description": "Cordless vacuum cleaner with laser"},
    {"title": "PlayStation 5", "description": "Gaming console with SSD"},
    {"title": "Nike Air Max 90", "description": "Running shoes white and black"},
    {"title": "Lego Star Wars Millennium Falcon", "description": "Building set 7541 pieces"},
]


def test_health():
    """Test health endpoint"""
    print("Testing /health...")
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        print(f"   Status: {r.status_code}")
        print(f"   Response: {r.json()}")
        return r.status_code == 200
    except Exception as e:
        print(f"   Error: {e}")
        return False


def test_classify(products):
    """Test classification endpoint with multiple products"""
    print(f"\nTesting /classify with {len(products)} products...")
    
    results = []
    for i, product in enumerate(products):
        try:
            start = time.time()
            r = requests.post(f"{API_URL}/classify", json=product, timeout=30)
            elapsed = (time.time() - start) * 1000
            
            if r.status_code == 200:
                data = r.json()
                cat_name = data.get('category_name', 'N/A')[:25]
                print(f"   [{i+1}] {product['title'][:30]:30} → {cat_name:25} (conf: {data['confidence']:.2f}, {elapsed:.0f}ms)")
                results.append(data)
            else:
                print(f"   [{i+1}] Error: {r.status_code} - {r.text}")
        except Exception as e:
            print(f"   [{i+1}] Error: {e}")
    
    return results


def main():
    print("="*60)
    print("API TEST SCRIPT")
    print(f"Target: {API_URL}")
    print("="*60)
    
    # Health check
    if not test_health():
        print("\nHealth check failed. Is the API running?")
        return
    
    # Classification tests
    results = test_classify(TEST_PRODUCTS)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"   Successful: {len(results)}/{len(TEST_PRODUCTS)}")
    if results:
        avg_conf = sum(r['confidence'] for r in results) / len(results)
        avg_time = sum(r['processing_time_ms'] for r in results) / len(results)
        print(f"   Avg confidence: {avg_conf:.2f}")
        print(f"   Avg latency: {avg_time:.0f}ms")


if __name__ == "__main__":
    main()

