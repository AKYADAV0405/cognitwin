import React, { createContext, useContext, useEffect, useState, useCallback } from "react";

export interface User {
  id: string;
  email: string;
  role: "student" | "researcher";
  auth_provider: "local" | "google";
  created_at: string;
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<User>;
  register: (email: string, password: string, role?: "student" | "researcher") => Promise<User>;
  logout: () => Promise<void>;
  refreshSession: () => Promise<boolean>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  // Silent token refresh helper
  const refreshSession = useCallback(async (): Promise<boolean> => {
    try {
      const res = await fetch("/api/v1/auth/refresh", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      return res.ok;
    } catch {
      return false;
    }
  }, []);

  const fetchCurrentUser = useCallback(async (retryOn401 = true): Promise<void> => {
    try {
      const res = await fetch("/api/v1/auth/me", {
        headers: { "Content-Type": "application/json" },
      });

      if (res.ok) {
        const userData: User = await res.json();
        setUser(userData);
      } else if (res.status === 401 && retryOn401) {
        // Attempt silent refresh once
        const refreshed = await refreshSession();
        if (refreshed) {
          const secondRes = await fetch("/api/v1/auth/me", {
            headers: { "Content-Type": "application/json" },
          });
          if (secondRes.ok) {
            const data: User = await secondRes.json();
            setUser(data);
            return;
          }
        }
        setUser(null);
      } else {
        setUser(null);
      }
    } catch {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, [refreshSession]);

  useEffect(() => {
    fetchCurrentUser();
  }, [fetchCurrentUser]);

  const login = async (email: string, password: string): Promise<User> => {
    const res = await fetch("/api/v1/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Invalid email or password.");
    }

    const userData: User = await res.json();
    setUser(userData);
    return userData;
  };

  const register = async (
    email: string,
    password: string,
    role: "student" | "researcher" = "student"
  ): Promise<User> => {
    const res = await fetch("/api/v1/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, role }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Registration failed.");
    }

    const userData: User = await res.json();
    setUser(userData);
    return userData;
  };

  const logout = async (): Promise<void> => {
    try {
      await fetch("/api/v1/auth/logout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
    } finally {
      setUser(null);
    }
  };

  return (
    <AuthContext.Provider
      value={{ user, loading, login, register, logout, refreshSession }}
    >
      {children}
    </AuthContext.Provider>
  );
};

// eslint-disable-next-line react-refresh/only-export-components
export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
