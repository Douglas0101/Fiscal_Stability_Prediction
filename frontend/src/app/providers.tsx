"use client";

import { SessionProvider } from "next-auth/react";
import React from "react";

// O componente Providers envolve a sua aplicação com o SessionProvider do NextAuth.
// Isto permite que qualquer componente filho aceda aos dados da sessão (ex: saber se o utilizador está logado).
export default function Providers({ children }: { children: React.ReactNode }) {
  return <SessionProvider>{children}</SessionProvider>;
}
