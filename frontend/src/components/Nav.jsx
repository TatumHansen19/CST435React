import { NavLink } from "react-router-dom";

const link = ({ isActive }) => (isActive ? "nav-link active" : "nav-link");

export default function Nav() {
  return (
    <nav className="nav">
      <div className="brand">CST Project Hub</div>
      <div className="links">
        <NavLink to="/" className={link} end>Home</NavLink>
        <NavLink to="/p1" className={link}>P1</NavLink>
        <NavLink to="/p2" className={link}>P2</NavLink>
        <NavLink to="/p3" className={link}>P3</NavLink>
        <NavLink to="/p4" className={link}>P4</NavLink>
        <NavLink to="/p5" className={link}>P5 (RNN)</NavLink>
        <NavLink to="/p6" className={link}>P6</NavLink>
        <NavLink to="/p7" className={link}>P7</NavLink>
        <NavLink to="/p8" className={link}>P8</NavLink>
      </div>
    </nav>
  );
}
