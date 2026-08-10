import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DocIntel",
  description: "Enterprise document intelligence workspace",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
