"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/auth";

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const token = useAuthStore((state) => state.token);
  useEffect(() => { if (token === null) router.replace("/login"); }, [token, router]);
  if (!token) return <main className="center-page"><p>Vérification de la session…</p></main>;
  return <>{children}</>;
}
