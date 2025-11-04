export default function Card({ title, subtitle, to }) {
  return (
    <a className="card" href={to}>
      <div className="card-title">{title}</div>
      <div className="card-sub">{subtitle}</div>
    </a>
  );
}
