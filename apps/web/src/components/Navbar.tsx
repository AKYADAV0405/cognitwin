import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { Activity, LogOut, ArrowRight, User } from "lucide-react";

export const Navbar: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    await logout();
    navigate("/");
  };

  return (
    <header className="sticky top-0 z-50 w-full h-16 bg-canvas/90 backdrop-blur-md border-b border-border-subtle">
      <div className="max-w-6xl mx-auto h-full px-6 flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center space-x-2.5 text-ink hover:opacity-90 transition-opacity">
          <div className="w-8 h-8 rounded-lg bg-ink flex items-center justify-center text-canvas shadow-sm">
            <Activity className="w-4 h-4 text-canvas" />
          </div>
          <span className="text-base font-bold tracking-tight text-ink">Cognitwin</span>
        </Link>

        {/* Center Nav Links */}
        <nav className="hidden md:flex items-center space-x-8 text-xs font-medium text-ink-body">
          <a href="/#overview" className="hover:text-ink transition-colors">Overview</a>
          <a href="/#how-it-works" className="hover:text-ink transition-colors">How It Works</a>
          <a href="/#extension" className="hover:text-ink transition-colors">Extension</a>
          <Link to="/research" className="hover:text-ink transition-colors flex items-center gap-1">
            Research
            <span className="text-[10px] font-mono px-1 py-0.2 rounded bg-surface-subtle text-ink-muted border border-border-subtle">Paper 1</span>
          </Link>
        </nav>

        {/* Right Actions */}
        <div className="flex items-center space-x-4">
          {user ? (
            <div className="flex items-center space-x-3">
              <Link
                to="/dashboard"
                className="flex items-center space-x-2 px-3 py-1.5 rounded-md bg-surface border border-border-subtle text-xs font-medium text-ink hover:bg-surface-subtle transition-colors"
              >
                <User className="w-3.5 h-3.5 text-cobalt" />
                <span className="truncate max-w-[120px]">{user.email.split("@")[0]}</span>
                <span className="text-[10px] uppercase font-mono px-1 rounded bg-surface-subtle text-ink-muted">
                  {user.role}
                </span>
              </Link>
              <button
                onClick={handleLogout}
                className="p-1.5 text-ink-muted hover:text-ink rounded-md hover:bg-surface transition-colors"
                title="Sign Out"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <div className="flex items-center space-x-3">
              <Link
                to="/login"
                className="text-xs font-medium text-ink-body hover:text-ink transition-colors px-2 py-1"
              >
                Log In
              </Link>
              <Link
                to="/signup"
                className="inline-flex items-center space-x-1.5 px-3.5 py-1.5 rounded-md bg-ink text-canvas text-xs font-medium hover:bg-ink-body transition-colors shadow-sm"
              >
                <span>Get Started</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
