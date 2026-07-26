"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { apiFetch, jsonBody, ApiError } from "@/lib/api";
import { AuthGuard } from "@/components/layout/AuthGuard";
import { MovieCard } from "@/components/movies/MovieCard";
import { useAuthStore } from "@/stores/auth";
import type { CatalogItem } from "@/types/api";

const genres = ["Action", "Aventure", "Comédie", "Drame", "Science-fiction", "Thriller", "Animation", "Documentaire"];

function OnboardingContent() {
  const router = useRouter(); const token = useAuthStore((state) => state.token); const [movies, setMovies] = useState<CatalogItem[]>([]); const [selectedGenres, setSelectedGenres] = useState<string[]>([]); const [favorites, setFavorites] = useState<number[]>([]); const [error, setError] = useState(""); const [loading, setLoading] = useState(false);
  useEffect(() => { if (token) apiFetch<CatalogItem[]>("/api/v1/catalog?item_type=movie&limit=8", {}, token).then(setMovies).catch(() => setMovies([])); }, [token]);
  const toggle = (value: string) => setSelectedGenres((current) => current.includes(value) ? current.filter((item) => item !== value) : [...current, value]);
  const toggleMovie = (id: number) => setFavorites((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id]);
  async function submit(event: React.FormEvent) { event.preventDefault(); if (!token) return; setLoading(true); try { await apiFetch("/api/v1/onboarding", { ...jsonBody({ preferred_genres: selectedGenres, favorite_movie_ids: favorites }) }, token); router.push("/movies"); } catch (exc) { setError(exc instanceof ApiError ? exc.detail : "Impossible d’enregistrer vos préférences."); } finally { setLoading(false); } }
  return <main className="page"><div className="hero"><div><p className="eyebrow">Première étape</p><h1>Parlez-nous de vous.</h1><p>Choisissez quelques genres et films que vous aimez pour personnaliser votre première sélection.</p></div></div><form onSubmit={submit}><section className="section"><h2>Vos genres préférés</h2><div className="chips">{genres.map((genre) => <button type="button" key={genre} className={`chip ${selectedGenres.includes(genre) ? "selected" : ""}`} onClick={() => toggle(genre)}>{genre}</button>)}</div></section><section className="section"><h2>Quelques films que vous aimez</h2><div className="movie-grid">{movies.map((movie) => <div key={movie.id} onClick={() => toggleMovie(movie.id)} className={favorites.includes(movie.id) ? "selected-movie" : ""}><MovieCard item={movie} /><p className="chip selected" style={{ margin: "-8px 10px 10px", textAlign: "center" }}>{favorites.includes(movie.id) ? "Sélectionné" : "Choisir"}</p></div>)}</div></section>{error && <p className="error">{error}</p>}<button className="button" disabled={loading}>{loading ? "Enregistrement…" : "Continuer vers les films"}</button></form></main>;
}

export default function OnboardingPage() { return <AuthGuard><OnboardingContent /></AuthGuard>; }
