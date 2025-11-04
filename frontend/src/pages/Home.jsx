import Card from "../components/Card.jsx";

const projects = [
  { id: 1, title: "Project 1: Perceptron", to: "/p1" },
  { id: 2, title: "Project 2: Artificial Neural Network (ANN)", to: "/p2" },
  { id: 3, title: "Project 3: Neural Network", to: "/p3" },
  { id: 4, title: "Project 4: NLP Application", to: "/p4" },
  { id: 5, title: "Project 5: Recurrent Neural Network (RNN)", to: "/p5" },
  { id: 6, title: "Project 6: Deep Neural Network Performance", to: "/p6" },
  { id: 7, title: "Project 7: GAN-Based Application", to: "/p7" },
  { id: 8, title: "Project 8: Deep Neural Network Project", to: "/p8" }
];

export default function Home() {
  return (
    <div className="grid">
      {projects.map(p => (
        <Card key={p.id} title={p.title} subtitle="Open" to={p.to} />
      ))}
    </div>
  );
}
