import type { Metadata } from "next";
import { Providers } from "./providers";
import { NavBar } from "@/components/layout/NavBar";
import "./globals.css";

export const metadata: Metadata = { title: "RecoSphere", description: "Vos films, mieux recommandés." };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="fr"><body><Providers><NavBar />{children}</Providers></body></html>;
}
