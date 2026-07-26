"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/auth";

export function NavBar() {
  const router = useRouter();
  const { user, clearSession } = useAuthStore();
  return (
    <header className="nav">
      <Link href="/movies" className="brand">Reco<span>Sphere</span></Link>
      <nav>
        <Link href="/movies">Films</Link>
        <Link href="/products">Produits</Link>
        <Link href="/recommendations">Pour vous</Link>
        <Link href="/favorites">Favoris</Link>
        {user?.is_admin && <Link href="/admin">Admin</Link>}
      </nav>
      <div className="nav-user">
        {user ? <><span>{user.display_name || user.email}</span><button onClick={() => { clearSession(); router.push("/login"); }}>Déconnexion</button></> : <Link href="/login">Connexion</Link>}
      </div>
    </header>
  );
}
