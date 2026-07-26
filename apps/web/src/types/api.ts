export type ItemType = "movie" | "product";

export interface User {
  id: number;
  email: string;
  display_name: string | null;
  preferred_language: string;
  country: string;
  is_admin: boolean;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

export interface CatalogItem {
  id: number;
  item_type: ItemType;
  title: string;
  description: string | null;
  image_url: string | null;
  detail_url: string | null;
  category: string | null;
  language: string | null;
  country: string | null;
  popularity_score: number;
  average_rating: number | null;
  rating_count: number;
  is_active: boolean;
  published_at: string | null;
  release_date?: string | null;
  runtime_minutes?: number | null;
  original_title?: string | null;
  original_language?: string | null;
  trailer_url?: string | null;
  genres?: string[];
  cast_members?: string[];
  directors?: string[];
  watch_providers?: Record<string, unknown>;
  price_amount?: number | null;
  price_currency?: string | null;
  condition?: string | null;
  condition_description?: string | null;
  brand?: string | null;
  seller_name?: string | null;
  availability?: string | null;
  shipping_cost?: number | null;
  shipping_currency?: string | null;
  marketplace?: string | null;
  product_url?: string | null;
  additional_images?: string[];
  item_end_date?: string | null;
}

export interface Recommendation {
  catalog_item_id: number;
  title: string;
  image_url: string | null;
  score: number;
  reason: string | null;
  components: Record<string, number>;
  item_type: ItemType;
  detail: CatalogItem;
}

export interface RecommendationResponse {
  user_id: number;
  item_type: ItemType;
  model_version: string;
  top_k: number;
  recommendations: Recommendation[];
}

export interface InteractionPayload {
  catalog_item_id: number;
  event_type: "impression" | "view" | "click" | "like" | "dislike" | "favorite" | "trailer_play" | "provider_click" | "purchase_redirect";
  event_value?: number;
  source_page?: string;
}

export interface AdminMetrics {
  users: number;
  movies: number;
  products: number;
  interactions: number;
  favorites: number;
  last_sync: Record<string, { status: string; started_at: string; finished_at: string | null; items_created: number; items_updated: number }>;
}
