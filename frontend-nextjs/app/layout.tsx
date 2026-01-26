import type { Metadata } from "next";
import React from "react";
import "./globals.css";
import { Navigation } from "@/components/Navigation";

export const metadata: Metadata = {
  title: "Ecommerce Classification API",
  description: "Interactive interface for testing e-commerce product classification",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col">
        <header className="sticky top-0 z-20 shrink-0">
          <Navigation />
        </header>
        <main className="flex-1 w-full">
          {children}
        </main>
      </body>
    </html>
  );
}

