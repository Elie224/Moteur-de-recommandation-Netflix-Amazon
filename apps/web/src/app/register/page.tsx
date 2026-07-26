"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { apiFetch, jsonBody, ApiError } from "@/lib/api";
import { useAuthStore } from "@/stores/auth";
import type { AuthResponse } from "@/types/api";

export default function RegisterPage() {
  const router = useRouter(); const setSession = useAuthStore((state) => state.setSession);
  const [name, setName] = useState(""); const [email, setEmail] = useState(""); const [password, setPassword] = useState(""); const [confirmation, setConfirmation] = useState(""); const [error, setError] = useState(""); const [loading, setLoading] = useState(false);
  async function submit(event: React.FormEvent) { event.preventDefault(); if (password !== confirmation) { setError("Les mots de passe ne correspondent pas."); return; } setError(""); setLoading(true); try { const result = await apiFetch<AuthResponse>("/api/v1/auth/register", jsonBody({ email, password, display_name: name || null })); setSession(result.access_token, result.user); router.push("/onboarding"); } catch (exc) { setError(exc instanceof ApiError ? exc.detail : "Inscription impossible."); } finally { setLoading(false); } }
  return <main className="form-page"><form className="form-card" onSubmit={submit}><p className="eyebrow">Commencez votre découverte</p><h1>Créer un compte</h1>{error && <p className="error">{error}</p>}<label>Nom<input value={name} onChange={(event) => setName(event.target.value)} placeholder="Votre prénom" /></label><label>Email<input required type="email" value={email} onChange={(event) => setEmail(event.target.value)} /></label><label>Mot de passe<input required minLength={8} type="password" value={password} onChange={(event) => setPassword(event.target.value)} /></label><label>Confirmation<input required minLength={8} type="password" value={confirmation} onChange={(event) => setConfirmation(event.target.value)} /></label><button className="button" disabled={loading}>{loading ? "Création…" : "Créer mon compte"}</button><p className="link-note">Déjà inscrit ? <Link href="/login">Se connecter</Link></p></form></main>;
}
