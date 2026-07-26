"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { apiFetch, jsonBody, ApiError } from "@/lib/api";
import { useAuthStore } from "@/stores/auth";
import type { AuthResponse } from "@/types/api";

export default function LoginPage() {
  const router = useRouter();
  const setSession = useAuthStore((state) => state.setSession);
  const [email, setEmail] = useState(""); const [password, setPassword] = useState(""); const [error, setError] = useState(""); const [loading, setLoading] = useState(false);
  async function submit(event: React.FormEvent) { event.preventDefault(); setError(""); setLoading(true); try { const result = await apiFetch<AuthResponse>("/api/v1/auth/login", jsonBody({ email, password })); setSession(result.access_token, result.user); router.push("/movies"); } catch (exc) { setError(exc instanceof ApiError ? exc.detail : "Connexion impossible."); } finally { setLoading(false); } }
  return <main className="form-page"><form className="form-card" onSubmit={submit}><p className="eyebrow">Bienvenue</p><h1>Connexion</h1>{error && <p className="error">{error}</p>}<label>Email<input required type="email" value={email} onChange={(event) => setEmail(event.target.value)} /></label><label>Mot de passe<input required type="password" value={password} onChange={(event) => setPassword(event.target.value)} /></label><button className="button" disabled={loading}>{loading ? "Connexion…" : "Se connecter"}</button><p className="link-note">Pas encore de compte ? <Link href="/register">Créer un compte</Link></p></form></main>;
}
