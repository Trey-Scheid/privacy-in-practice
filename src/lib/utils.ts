export function getPublicPath(path: string): string {
  // Ensure path doesn't have leading slash for consistency
  const cleanPath = path.startsWith('/') ? path.slice(1) : path;

  // Check if we're running on GitHub Pages
  const isGitHubPages = typeof window !== 'undefined' && window.location.hostname.includes('github.io');
  
  if (isGitHubPages) {
    // Get repository name from URL
    const repoName = window.location.pathname.split('/')[1];
    return `/${repoName}/${cleanPath}`;
  } else {
    // Local development
    return `/${cleanPath}`;
  }
} 