export interface Summary {
  available: boolean;
  total_customers?: number;
  avg_churn_probability?: number;
  high_risk_count?: number;
  risk_distribution?: Record<string, number>;
}

export interface CustomerProfile {
  customer_id: string;
  name: string;
  email: string;
  country: string;
  age: number;
  age_group: string;
  clv_tier: string;
  signup_date: string | null;
  first_session_date: string | null;
  total_orders: number;
  total_spend_usd: number;
  avg_order_value: number | null;
  avg_discount_pct: number | null;
  total_sessions: number;
  has_abandoned_cart: number;
  marketing_opt_in: boolean;
  is_repeat_customer: number;
  top_category_bought: string | null;
  preferred_device_ord: string | null;
  preferred_source: string | null;
  preferred_payment: string | null;
  preferred_device_sess: string | null;
  preferred_source_sess: string | null;
  avg_rating_given: number | null;
  last_order_date: string | null;
  last_session_date: string | null;
  churn_probability: number;
  risk_tier: string;
  risk_explanation: string;
  segment: string;
  predicted_clv: number | null;
  predicted_clv_tier: string;
  predicted_orders_12m: number | null;
  cart_abandon_probability: number | null;
  discount_sensitivity: number | null;
}

export interface ChatResponse {
  reply: string;
}

export interface Insight {
  title: string;
  text: string;
}

export interface InsightsResponse {
  available: boolean;
  insights: Insight[];
}

export interface Population {
  available: boolean;
  total_customers?: number;
  age_avg?: number;
  total_orders_avg?: number;
  total_orders_sum?: number;
  total_spend_sum?: number;
  total_spend_avg?: number;
  avg_order_value_avg?: number;
  avg_discount_pct_avg?: number;
  predicted_clv_avg?: number | null;
  segment_top?: string;
  clv_tier_top?: string;
  repeat_rate?: number;
  total_sessions_avg?: number;
  device_sess_top?: string;
  source_sess_top?: string;
  cart_abandon_rate?: number;
  cart_abandon_prob_avg?: number | null;
  payment_top?: string;
  device_ord_top?: string;
  source_ord_top?: string;
  top_category?: string;
  avg_rating_avg?: number;
  discount_sensitivity_avg?: number | null;
}

export interface ChatMessage {
  role: "user" | "bot";
  text: string;
}
