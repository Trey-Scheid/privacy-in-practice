export function getPublicPath(path: string): string {
  // Check if we're running on GitHub Pages
  const isGitHubPages = typeof window !== 'undefined' && window.location.hostname.includes('github.io');
  
  // Get repository name from URL if on GitHub Pages
  const basePath = isGitHubPages 
    ? '/' + window.location.pathname.split('/')[1]
    : '';
    
  return `${basePath}${path}`;
} 