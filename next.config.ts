import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  basePath: process.env.NODE_ENV === 'production' 
    ? '/privacy-in-practice'
    : '',
  output: 'export',
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
