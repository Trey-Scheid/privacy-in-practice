export function getPublicPath(path: string): string {
  // Check if we're running on GitHub Pages
  const isGitHubPages = typeof window !== 'undefined' && window.location.hostname.includes('github.io');
  
  // Get repository name from URL if on GitHub Pages
  const basePath = isGitHubPages 
    ? '/' + window.location.pathname.split('/')[1]
    : '';
    
  // Ensure path starts with a forward slash and remove any double slashes
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${basePath}${normalizedPath}`.replace(/\/+/g, '/');
} 