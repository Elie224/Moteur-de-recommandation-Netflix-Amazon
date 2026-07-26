"use client";

import { useQuery } from "@tanstack/react-query";
import { AuthGuard } from "@/components/layout/AuthGuard";
import { MovieCard } from "@/components/movies/MovieCard";
import { apiFetch } from "@/lib/api";
import { useAuthStore } from "@/stores/auth";
import type { RecommendationResponse } from "@/types/api";

function RecommendationsContent() { const token = useAuthStore((state) => state.token); const query = useQuery({ queryKey: ["recommendations", "movie", "full"], queryFn: () => apiFetch<RecommendationResponse>("/api/v1/recommendations/movie?top_k=50", {}, token!), enabled: Boolean(token) }); if (query.isLoading) return <main className="center-page"><p>Calcul de vos recommandations…</p></main>; return <main className="page"><div className="hero"><div><p className="eyebrow">Sur mesure</p><h1>Pour vous.</h1><p>Les recommandations combinent popularité, contenu, activité récente et similarité.</p></div></div>{query.error ? <p className="error">Impossible de charger les recommandations.</p> : <div className="movie-grid">{(query.data?.recommendations ?? []).map((item) => <MovieCard key={item.catalog_item_id} item={item.detail} reason={item.reason} />)}</div>}</main>; }
export default function RecommendationsPage() { return <AuthGuard><RecommendationsContent /></AuthGuard>; }
