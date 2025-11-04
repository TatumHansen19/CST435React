import { Routes, Route } from "react-router-dom";
import Nav from "./components/Nav.jsx";
import Home from "./pages/Home.jsx";
import P1 from "./pages/P1.jsx";
import P2 from "./pages/P2.jsx";
import P3 from "./pages/P3.jsx";
import P4 from "./pages/P4.jsx";
import P5_RNN from "./pages/P5.jsx";
import P6 from "./pages/P6.jsx";
import P7 from "./pages/P7.jsx";
import P8 from "./pages/P8.jsx";

export default function App() {
  return (
    <div className="app">
      <Nav />
      <div className="content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/p1" element={<P1 />} />
          <Route path="/p2" element={<P2 />} />
          <Route path="/p3" element={<P3 />} />
          <Route path="/p4" element={<P4 />} />
          <Route path="/p5" element={<P5_RNN />} />
          <Route path="/p6" element={<P6 />} />
          <Route path="/p7" element={<P7 />} />
          <Route path="/p8" element={<P8 />} />
        </Routes>
      </div>
    </div>
  );
}
