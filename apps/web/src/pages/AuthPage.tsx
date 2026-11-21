import React, { useState, useEffect } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { KnowledgeGraphPreview } from "../components/KnowledgeGraphPreview";
import {
  Activity,
  Eye,
  EyeOff,
  AlertCircle,
  CheckCircle2,
  Lock,
  Mail,
  GraduationCap,
  Microscope,
  ArrowRight,
  ShieldCheck,
} from "lucide-react";

export const AuthPage: React.FC = () => {
  const { login, register } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const isSignUpRoute = location.pathname === "/signup";
  const [isSignUp, setIsSignUp] = useState<boolean>(isSignUpRoute);

  useEffect(() => {
    setIsSignUp(location.pathname === "/signup");
    setError(null);
  }, [location.pathname]);

  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [role, setRole] = useState<"student" | "researcher">("student");
  const [showPassword, setShowPassword] = useState<boolean>(false);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Live password strength criteria
  const hasLength = password.length >= 10;
  const hasUpper = /[A-Z]/.test(password);
  const hasLower = /[a-z]/.test(password);
  const hasNumber = /\d/.test(password);
  const hasSpecial = /[!@#$%^&*(),.?":{}|<>\-_=+[\];/]/.test(password);
  const passedCount = [hasLength, hasUpper, hasLower, hasNumber, hasSpecial].filter(Boolean).length;

  const getStrengthLabel = () => {
    if (passedCount <= 2) return { text: "Weak", color: "bg-status-risk", width: "w-1/4" };
    if (passedCount <= 4) return { text: "Moderate", color: "bg-status-fading", width: "w-3/4" };
    return { text: "Strong", color: "bg-status-mastered", width: "w-full" };
  };

  const strength = getStrengthLabel();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (isSignUp) {
        if (passedCount < 5) {
          throw new Error("Please fulfill all password security requirements before creating an account.");
        }
        await register(email, password, role);
      } else {
        await login(email, password);
      }
      navigate("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleLogin = () => {
    window.location.href = "/api/v1/auth/google/login";
  };

  return (
    <div className="min-h-screen bg-canvas grid grid-cols-1 lg:grid-cols-12 text-ink">
      {/* ==================================================================== */}
      {/* Left Column: Authentic White Form Panel */}
      {/* ==================================================================== */}
      <div className="lg:col-span-6 flex flex-col justify-between p-6 sm:p-12 lg:p-16 bg-canvas border-r border-border-subtle">
        {/* Brand Header */}
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center space-x-2 text-ink">
            <div className="w-8 h-8 rounded-lg bg-ink flex items-center justify-center text-canvas shadow-sm">
              <Activity className="w-4 h-4 text-canvas" />
            </div>
            <span className="text-base font-bold tracking-tight text-ink">Cognitwin</span>
          </Link>
          <Link
            to="/"
            className="text-xs text-ink-muted hover:text-ink transition-colors font-medium"
          >
            &larr; Back to Home
          </Link>
        </div>

        {/* Central Auth Card Form */}
        <div className="max-w-md w-full mx-auto my-8 space-y-6">
          <div>
            <h1 className="font-editorial text-3xl font-medium tracking-tight text-ink mb-1.5">
              {isSignUp ? "Create your cognitive twin." : "Welcome back."}
            </h1>
            <p className="text-xs text-ink-body leading-relaxed">
              {isSignUp
                ? "Start mapping your technical knowledge with dual-signal telemetry."
                : "Sign in to access your real-time Knowledge Graph and review queues."}
            </p>
          </div>

          {/* Segmented Switcher */}
          <div className="flex bg-surface p-1 rounded-lg border border-border-subtle text-xs font-medium">
            <button
              type="button"
              onClick={() => {
                setIsSignUp(false);
                setError(null);
              }}
              className={`flex-1 py-1.5 rounded-md transition-all ${
                !isSignUp
                  ? "bg-canvas text-ink shadow-sm font-semibold"
                  : "text-ink-muted hover:text-ink"
              }`}
            >
              Sign In
            </button>
            <button
              type="button"
              onClick={() => {
                setIsSignUp(true);
                setError(null);
              }}
              className={`flex-1 py-1.5 rounded-md transition-all ${
                isSignUp
                  ? "bg-canvas text-ink shadow-sm font-semibold"
                  : "text-ink-muted hover:text-ink"
              }`}
            >
              Create Account
            </button>
          </div>

          {/* Error / Rate-Limit Alert Banner */}
          {error && (
            <div className="p-3 rounded-lg bg-status-risk-bg border border-status-risk-border flex items-start space-x-2.5 text-xs text-status-risk">
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <div className="flex-1 leading-relaxed">{error}</div>
            </div>
          )}

          {/* Google OAuth Button */}
          <button
            type="button"
            onClick={handleGoogleLogin}
            className="w-full flex items-center justify-center space-x-2.5 px-4 py-2.5 rounded-lg border border-border-subtle bg-canvas hover:bg-surface text-xs font-medium text-ink transition-all shadow-sm"
          >
            {/* Google SVG G-Logo */}
            <svg className="w-4 h-4" viewBox="0 0 24 24">
              <path
                fill="#4285F4"
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
              />
              <path
                fill="#34A853"
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
              />
              <path
                fill="#FBBC05"
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
              />
              <path
                fill="#EA4335"
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
              />
            </svg>
            <span>Continue with Google</span>
          </button>

          {/* Divider */}
          <div className="relative flex items-center justify-center">
            <div className="border-t border-border-subtle w-full" />
            <span className="bg-canvas px-3 text-[11px] font-mono text-ink-subtle uppercase">
              or credentials
            </span>
          </div>

          {/* Credentials Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Email Field */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-ink flex items-center justify-between">
                <span>Institutional or Personal Email</span>
              </label>
              <div className="relative">
                <Mail className="w-4 h-4 text-ink-muted absolute left-3 top-2.5 pointer-events-none" />
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="alex@university.edu"
                  className="w-full pl-9 pr-3 py-2 rounded-lg border border-border-subtle bg-canvas text-xs text-ink placeholder:text-ink-subtle focus:outline-none focus:border-cobalt focus:ring-1 focus:ring-cobalt transition-all"
                />
              </div>
            </div>

            {/* Password Field */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <label className="font-medium text-ink">Password</label>
                {!isSignUp && (
                  <span className="text-[11px] text-cobalt hover:underline cursor-pointer">
                    Forgot password?
                  </span>
                )}
              </div>
              <div className="relative">
                <Lock className="w-4 h-4 text-ink-muted absolute left-3 top-2.5 pointer-events-none" />
                <input
                  type={showPassword ? "text" : "password"}
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••••••"
                  className="w-full pl-9 pr-10 py-2 rounded-lg border border-border-subtle bg-canvas text-xs text-ink placeholder:text-ink-subtle focus:outline-none focus:border-cobalt focus:ring-1 focus:ring-cobalt transition-all font-mono"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-2.5 text-ink-muted hover:text-ink transition-colors"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* Live Password Strength Meter (On Sign Up) */}
            {isSignUp && password.length > 0 && (
              <div className="space-y-2 p-3 bg-surface rounded-lg border border-border-subtle text-xs">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="font-medium text-ink-body">Password Complexity:</span>
                  <span className="font-mono font-semibold text-ink">{strength.text}</span>
                </div>
                <div className="w-full h-1.5 bg-border-subtle rounded-full overflow-hidden">
                  <div className={`h-full ${strength.color} ${strength.width} transition-all duration-300`} />
                </div>
                <div className="grid grid-cols-2 gap-1 text-[10px] text-ink-muted pt-1">
                  <span className={hasLength ? "text-status-mastered font-medium" : ""}>
                    {hasLength ? "✓" : "○"} 10+ characters
                  </span>
                  <span className={hasUpper ? "text-status-mastered font-medium" : ""}>
                    {hasUpper ? "✓" : "○"} Uppercase letter
                  </span>
                  <span className={hasLower ? "text-status-mastered font-medium" : ""}>
                    {hasLower ? "✓" : "○"} Lowercase letter
                  </span>
                  <span className={hasNumber ? "text-status-mastered font-medium" : ""}>
                    {hasNumber ? "✓" : "○"} Numeric digit
                  </span>
                  <span className={hasSpecial ? "text-status-mastered font-medium" : ""}>
                    {hasSpecial ? "✓" : "○"} Special character
                  </span>
                </div>
              </div>
            )}

            {/* Role Selector (On Sign Up) */}
            {isSignUp && (
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-ink">Workspace Role</label>
                <div className="grid grid-cols-2 gap-3">
                  <div
                    onClick={() => setRole("student")}
                    className={`cursor-pointer p-3 rounded-lg border text-left transition-all ${
                      role === "student"
                        ? "border-cobalt bg-cobalt-surface/40 ring-1 ring-cobalt"
                        : "border-border-subtle bg-canvas hover:bg-surface"
                    }`}
                  >
                    <div className="flex items-center space-x-1.5 text-xs font-semibold text-ink mb-1">
                      <GraduationCap className="w-3.5 h-3.5 text-cobalt" />
                      <span>Student</span>
                    </div>
                    <p className="text-[10px] text-ink-muted leading-tight">
                      Mastery graph & 2-minute verification quizzes.
                    </p>
                  </div>

                  <div
                    onClick={() => setRole("researcher")}
                    className={`cursor-pointer p-3 rounded-lg border text-left transition-all ${
                      role === "researcher"
                        ? "border-cobalt bg-cobalt-surface/40 ring-1 ring-cobalt"
                        : "border-border-subtle bg-canvas hover:bg-surface"
                    }`}
                  >
                    <div className="flex items-center space-x-1.5 text-xs font-semibold text-ink mb-1">
                      <Microscope className="w-3.5 h-3.5 text-cobalt" />
                      <span>Researcher</span>
                    </div>
                    <p className="text-[10px] text-ink-muted leading-tight">
                      Paper 1 ROC-AUC, RMSE benchmarks & model weights.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Primary Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center space-x-2 px-4 py-2.5 rounded-lg bg-cobalt hover:bg-cobalt-hover text-canvas text-xs font-semibold transition-all disabled:opacity-50 shadow-sm"
            >
              <span>
                {loading
                  ? "Securing Session..."
                  : isSignUp
                  ? role === "researcher"
                    ? "Create Researcher Account"
                    : "Create Student Account"
                  : "Sign In to Cognitwin"}
              </span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </form>
        </div>

        {/* Security Footnote */}
        <div className="text-center text-[11px] text-ink-subtle flex items-center justify-center space-x-1">
          <ShieldCheck className="w-3.5 h-3.5 text-status-mastered" />
          <span>Protected by Argon2id, HTTP-only SameSite cookies & PKCE OAuth</span>
        </div>
      </div>

      {/* ==================================================================== */}
      {/* Right Column: Stitch Digital Twin Knowledge Graph Showcase Panel */}
      {/* ==================================================================== */}
      <div className="hidden lg:flex lg:col-span-6 bg-surface relative flex-col justify-center items-center p-12 overflow-hidden border-l border-border-subtle">
        {/* Wireframe Mesh Background */}
        <div className="absolute inset-0 bg-mesh-pattern pointer-events-none opacity-80" />

        <div className="relative z-10 w-full max-w-lg space-y-6">
          <div className="space-y-1.5 text-center">
            <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-cobalt">
              COGNITIVE DIGITAL TWIN
            </span>
            <h3 className="font-editorial text-2xl font-medium tracking-tight text-ink">
              Adaptive Knowledge Tracing in Real Time
            </h3>
            <p className="text-xs text-ink-body max-w-sm mx-auto leading-relaxed">
              Every passive study dwell session and active ground-truth quiz updates concept edge weights
              and forgetting decay curves.
            </p>
          </div>

          {/* Floating Showcase Preview Card */}
          <KnowledgeGraphPreview />

          <div className="bg-canvas border border-border-subtle rounded-lg p-4 shadow-sm text-xs text-ink-body flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <CheckCircle2 className="w-4 h-4 text-status-mastered" />
              <span className="font-medium text-ink">Strict Zero DOM Ingestion Guarantee</span>
            </div>
            <span className="text-[11px] font-mono text-ink-muted">Privacy First</span>
          </div>
        </div>
      </div>
    </div>
  );
};
