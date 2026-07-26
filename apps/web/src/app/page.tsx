import Link from "next/link";

export default function Home() {
  return <main className="center-page"><div className="form-card"><p className="eyebrow">RecoSphere</p><h1>Les bons films, au bon moment.</h1><p>Découvrez un catalogue enrichi par TMDB et des recommandations qui apprennent de vos goûts.</p><div className="detail-actions"><Link className="button" href="/register">Créer un compte</Link><Link className="button secondary" href="/login">Se connecter</Link></div></div></main>;
}
