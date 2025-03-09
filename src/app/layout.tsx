import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import "katex/dist/katex.min.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Privacy in Practice",
  description: "The Feasibility of Differential Privacy for Telemetry Analysis",
  openGraph: {
    title: "Privacy in Practice",
    description:
      "The Feasibility of Differential Privacy for Telemetry Analysis",
    type: "website",
    url: "https://trey-scheid.github.io/privacy-in-practice/",
    images: [
      {
        url: "og-image.png",
        width: 1200,
        height: 630,
        alt: "Privacy in Practice - Differential Privacy Research",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Privacy in Practice",
    description:
      "The Feasibility of Differential Privacy for Telemetry Analysis",
    images: ["/og-image.png"],
  },
  authors: [
    { name: "Tyler Kurpanek" },
    { name: "Chris Lum" },
    { name: "Bradley Nathanson" },
    { name: "Trey Scheid" },
  ],
  keywords: [
    "Differential Privacy",
    "Privacy",
    "Data Analysis",
    "Telemetry",
    "Research",
  ],
  icons: {
    icon: [
      { url: "/favicon.ico" },
      { url: "/favicon-32x32.png", sizes: "32x32", type: "image/png" },
      { url: "/apple-touch-icon.png", sizes: "180x180", type: "image/png" },
    ],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
