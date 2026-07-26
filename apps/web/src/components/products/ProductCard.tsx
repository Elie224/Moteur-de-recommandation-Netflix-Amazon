"use client";

import Link from "next/link";
import type { CatalogItem } from "@/types/api";

export function ProductCard({ item, onLike, onDislike, onFavorite }: { item: CatalogItem; onLike?: (id: number) => void; onDislike?: (id: number) => void; onFavorite?: (id: number) => void }) {
  return <article className="movie-card"><Link href={`/products/${item.id}`} className="poster-link">{item.image_url ? <img src={item.image_url} alt={`Image de ${item.title}`} /> : <div className="poster-placeholder">🛍️</div>}</Link><div className="movie-card-body"><div className="movie-card-title"><Link href={`/products/${item.id}`}><h3>{item.title}</h3></Link></div>{item.price_amount != null && <p className="rating">{item.price_amount.toFixed(2)} {item.price_currency ?? "EUR"}</p>}{item.brand && <p className="reason">{item.brand} · {item.condition ?? "État non précisé"}</p>} {(onLike || onDislike || onFavorite) && <div className="card-actions">{onLike && <button onClick={() => onLike(item.id)} aria-label={`Aimer ${item.title}`}>♥</button>}{onDislike && <button onClick={() => onDislike(item.id)} aria-label={`Ne pas aimer ${item.title}`}>✕</button>}{onFavorite && <button onClick={() => onFavorite(item.id)} aria-label={`Ajouter ${item.title} aux favoris`}>☆</button>}</div>}</div></article>;
}
