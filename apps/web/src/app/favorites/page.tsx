"use client";

import { useQuery } from "@tanstack/react-query";
import { AuthGuard } from "@/components/layout/AuthGuard";
import { MovieCard } from "@/components/movies/MovieCard";
import { apiFetch } from "@/lib/api";
import { useAuthStore } from "@/stores/auth";
import type { CatalogItem } from "@/types/api";

function FavoritesContent() { const token = useAuthStore((state) => state.token); const query = useQuery({ queryKey: ["favorites"], queryFn: () => apiFetch<CatalogItem[]>("/api/v1/favorites", {}, token!), enabled: Boolean(token) }); if (query.isLoading) return <main className="center-page"><p>Chargement de vos favoris…</p></main>; const movies = (query.data ?? []).filter((item) => item.item_type === "movie"); return <main className="page"><div className="hero"><div><p className="eyebrow">Votre collection</p><h1>Favoris.</h1><p>Retrouvez ici les films que vous voulez garder sous la main.</p></div></div>{query.error ? <p className="error">Impossible de charger vos favoris.</p> : movies.length ? <div className="movie-grid">{movies.map((item) => <MovieCard key={item.id} item={item} />)}</div> : <p className="empty">Vous n’avez pas encore de film favori.</p>}</main>; }
export default function FavoritesPage() { return <AuthGuard><FavoritesContent /></AuthGuard>; }
