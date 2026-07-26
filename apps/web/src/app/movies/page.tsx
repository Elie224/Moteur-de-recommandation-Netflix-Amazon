"use client";

import Link from "next/link";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { AuthGuard } from "@/components/layout/AuthGuard";
import { MovieCard } from "@/components/movies/MovieCard";
import { apiFetch, jsonBody, ApiError } from "@/lib/api";
import { useAuthStore } from "@/stores/auth";
import type { CatalogItem, InteractionPayload, RecommendationResponse } from "@/types/api";

function MoviesContent() {
  const token = useAuthStore((state) => state.token); const queryClient = useQueryClient();
  const catalog = useQuery({ queryKey: ["catalog", "movie"], queryFn: () => apiFetch<CatalogItem[]>("/api/v1/catalog?item_type=movie&limit=50", {}, token!), enabled: Boolean(token) });
  const recommendations = useQuery({ queryKey: ["recommendations", "movie"], queryFn: () => apiFetch<RecommendationResponse>("/api/v1/recommendations/movie?top_k=12", {}, token!), enabled: Boolean(token) });
  const interaction = useMutation({ mutationFn: (payload: InteractionPayload) => apiFetch("/api/v1/interactions", { ...jsonBody(payload) }, token!), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["recommendations", "movie"] }) });
  const favorite = useMutation({ mutationFn: (id: number) => apiFetch<{ is_favorite: boolean }>(`/api/v1/favorites/${id}`, { ...jsonBody({}) }, token!), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["favorites"] }) });
  const send = (id: number, event_type: InteractionPayload["event_type"]) => interaction.mutate({ catalog_item_id: id, event_type, source_page: "/movies" });
  if (catalog.isLoading || recommendations.isLoading) return <main className="center-page"><p>Chargement de votre sélection…</p></main>;
  if (catalog.error || recommendations.error) return <main className="center-page"><div className="form-card"><h1>Catalogue indisponible</h1><p className="error">{(catalog.error instanceof ApiError && catalog.error.detail) || "Impossible de charger les films."}</p><Link href="/login" className="button">Réessayer</Link></div></main>;
  const recs = recommendations.data?.recommendations ?? []; const movies = catalog.data ?? [];
  return <main className="page"><div className="hero"><div><p className="eyebrow">Votre cinéma personnel</p><h1>Films à découvrir.</h1><p>Chaque interaction affine votre sélection. Aimez ou rejetez un film pour faire évoluer vos recommandations.</p></div><Link className="button secondary" href="/onboarding">Modifier mes goûts</Link></div><section className="section"><h2>Recommandés pour vous</h2>{recs.length ? <div className="movie-grid">{recs.map((rec) => <MovieCard key={rec.catalog_item_id} item={rec.detail} reason={rec.reason} onLike={(id) => send(id, "like")} onDislike={(id) => send(id, "dislike")} onFavorite={(id) => favorite.mutate(id)} />)}</div> : <p className="empty">Aucune recommandation pour le moment. Synchronisez le catalogue ou ajoutez des préférences.</p>}</section><section className="section"><h2>Films populaires</h2>{movies.length ? <div className="movie-grid">{movies.map((movie) => <MovieCard key={movie.id} item={movie} onLike={(id) => send(id, "like")} onDislike={(id) => send(id, "dislike")} onFavorite={(id) => favorite.mutate(id)} />)}</div> : <p className="empty">Le catalogue est vide. Un administrateur doit lancer la synchronisation TMDB.</p>}</section></main>;
}

export default function MoviesPage() { return <AuthGuard><MoviesContent /></AuthGuard>; }
