import type {
  ChatResponse,
  CustomerProfile,
  InsightsResponse,
  Population,
  Summary,
} from "./types";

async function getJSON<T>(url: string): Promise<T> {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`${url} -> ${resp.status}`);
  }
  return (await resp.json()) as T;
}

async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    throw new Error(`${url} -> ${resp.status}`);
  }
  return (await resp.json()) as T;
}

export const getSummary = () => getJSON<Summary>("/api/summary");

export const getCustomer = (customerId: string) =>
  getJSON<CustomerProfile>(`/api/customers/${encodeURIComponent(customerId)}`);

export const getInsights = () => getJSON<InsightsResponse>("/api/insights");

export const getPopulation = () => getJSON<Population>("/api/population");

export const sendChat = (message: string, customerId: string | null) =>
  postJSON<ChatResponse>("/api/chat", { message, customer_id: customerId });
