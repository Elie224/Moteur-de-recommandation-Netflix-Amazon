"use client";

import { useMutation } from "@tanstack/react-query";
import { apiFetch, ApiError } from "@/lib/api";
import { useAuthStore } from "@/stores/auth";

export function EbaySyncButton() {
  const token = useAuthStore((state) => state.token);
  const sync = useMutation({ mutationFn: () => apiFetch<{ status: string; created: number; updated: number; failed: number }>("/api/v1/admin/sync/ebay", { method: "POST" }, token!), });
  return <div><button className="button secondary" disabled={sync.isPending} onClick={() => sync.mutate()}>{sync.isPending ? "Synchronisation eBay…" : "Synchroniser eBay"}</button>{sync.data && <p className="reason">eBay : {sync.data.created} créés, {sync.data.updated} mis à jour, {sync.data.failed} erreurs.</p>}{sync.error && <p className="error">{sync.error instanceof ApiError ? sync.error.detail : "La synchronisation eBay a échoué."}</p>}</div>;
}
