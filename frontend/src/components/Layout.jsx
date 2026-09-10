import { Outlet } from 'react-router-dom';
import Navbar from './Navbar';

export default function Layout() {
  return (
    <div className="app-layout">
      <Navbar />
      <div className="canvas-frame">
        <div className="canvas-background">
          <div className="bg-blur-layer" aria-hidden="true" />
          <div className="bg-gradient-veil" aria-hidden="true" />
          <main className="page-container">
            <Outlet />
          </main>
        </div>
      </div>
    </div>
  );
}
