"use client";

import Link from "next/link";
import type { CatalogItem } from "@/types/api";

interface MovieCardProps {
  item: CatalogItem;
  reason?: string | null;
  onLike?: (id: number) => void;
  onDislike?: (id: number) => void;
  onFavorite?: (id: number) => void;
}

export function MovieCard({ item, reason, onLike, onDislike, onFavorite }: MovieCardProps) {
  return (
    <article className="movie-card">
      <Link href={`/movies/${item.id}`} className="poster-link">
        {item.image_url ? <img src={item.image_url} alt={`Affiche du film ${item.title}`} /> : <div className="poster-placeholder">🎬</div>}
      </Link>
      <div className="movie-card-body">
        <div className="movie-card-title">
          <Link href={`/movies/${item.id}`}><h3>{item.title}</h3></Link>
          {item.release_date && <span>{new Date(item.release_date).getFullYear()}</span>}
        </div>
        {item.average_rating ? <p className="rating">★ {item.average_rating.toFixed(1)} <small>({item.rating_count})</small></p> : null}
        {reason && <p className="reason">{reason}</p>}
        {(onLike || onDislike || onFavorite) && (
          <div className="card-actions">
            {onLike && <button aria-label={`Aimer ${item.title}`} onClick={() => onLike(item.id)}>♥</button>}
            {onDislike && <button aria-label={`Ne pas aimer ${item.title}`} onClick={() => onDislike(item.id)}>✕</button>}
            {onFavorite && <button aria-label={`Ajouter ${item.title} aux favoris`} onClick={() => onFavorite(item.id)}>☆</button>}
          </div>
        )}
      </div>
    </article>
  );
}
