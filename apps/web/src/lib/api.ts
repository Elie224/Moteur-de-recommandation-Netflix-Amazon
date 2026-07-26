export const API_URL = typeof window === "undefined"
  ? process.env.RECOSPHERE_INTERNAL_API_URL ?? "http://api:8000"
  : process.env.NEXT_PUBLIC_RECOSPHERE_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(public readonly status: number, public readonly detail: string) {
    super(detail);
  }
}

export async function apiFetch<T>(path: string, init: RequestInit = {}, token?: string): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${API_URL}${path}`, { ...init, headers, cache: "no-store" });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new ApiError(response.status, payload?.detail ?? "Une erreur est survenue.");
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const jsonBody = (value: unknown): RequestInit => ({
  method: "POST",
  body: JSON.stringify(value),
});
