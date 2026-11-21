import React from "react";
import { Navigate, Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { ShieldAlert, ArrowLeft, LogOut } from "lucide-react";

interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredRole?: "student" | "researcher";
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requiredRole = "student",
}) => {
  const { user, loading, logout } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen bg-canvas flex items-center justify-center">
        <div className="flex flex-col items-center space-y-3">
          <div className="w-6 h-6 border-2 border-cobalt border-t-transparent rounded-full animate-spin" />
          <span className="text-xs font-mono text-ink-muted uppercase tracking-wider">
            Authenticating Session...
          </span>
        </div>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  // Enforce Researcher role if required
  if (requiredRole === "researcher" && user.role !== "researcher") {
    return (
      <div className="min-h-screen bg-surface flex items-center justify-center p-6">
        <div className="w-full max-w-md bg-canvas border border-border-subtle rounded-lg p-8 shadow-sm text-center">
          <div className="w-12 h-12 rounded-full bg-status-risk-bg border border-status-risk-border flex items-center justify-center mx-auto mb-4 text-status-risk">
            <ShieldAlert className="w-6 h-6" />
          </div>

          <h2 className="text-lg font-semibold text-ink tracking-tight mb-2">
            Researcher Authorization Required
          </h2>

          <p className="text-xs text-ink-body leading-relaxed mb-6">
            This module contains aggregate evaluation benchmarks, AUC/RMSE telemetry distributions,
            and neural graph checkpoints restricted to validated researchers. Your account is
            registered as a <span className="font-semibold text-ink">Student</span>.
          </p>

          <div className="space-y-2">
            <Link
              to="/dashboard"
              className="inline-flex items-center justify-center w-full px-4 py-2 rounded-md bg-ink text-canvas text-xs font-medium hover:bg-ink-body transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5 mr-2" />
              Return to Student Dashboard
            </Link>

            <button
              onClick={() => logout()}
              className="inline-flex items-center justify-center w-full px-4 py-2 rounded-md border border-border-subtle text-ink-body text-xs font-medium hover:bg-surface transition-colors"
            >
              <LogOut className="w-3.5 h-3.5 mr-2" />
              Sign Out & Switch Account
            </button>
          </div>
        </div>
      </div>
    );
  }

  return <>{children}</>;
};
