export interface Product {
  product_id?: string;
  title: string;
  description?: string;
  category_id?: string;
  category_name?: string;
  category_path?: string;
}

export interface ClassificationResult {
  category_id: string;
  category_name: string;
  category_path: string;
  confidence: number;
}

export interface PredictionResult extends Product {
  predicted_category_id: string;
  predicted_category_name: string;
  predicted_category_path: string;
  confidence: number;
  is_correct?: boolean;
  latency_ms: number;
}

export interface CategoryInfo {
  name: string;
  example_titles: string[];
}

export interface CategoryData {
  [categoryId: string]: CategoryInfo;
}

export interface Stats {
  total_tested: number;
  accuracy: number;
  avg_confidence: number;
  avg_latency_ms: number;
}


