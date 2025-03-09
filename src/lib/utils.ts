export function getPublicPath(path: string): string {
  if (!path) {
    console.error("Empty path provided to getPublicPath");
    return "";
  }

  // Remove leading slash if present
  const cleanPath = path.startsWith('/') ? path.slice(1) : path;
  
  // Check if we're running in the browser
  const isClient = typeof window !== 'undefined';
  
  // Detect GitHub Pages hosting
  const isGitHubPages = isClient && window.location.hostname.includes('github.io');
  
  // Build the appropriate path
  let finalPath;
  
  if (isGitHubPages) {
    // GitHub Pages deployment
    const repoName = window.location.pathname.split('/')[1];
    finalPath = `/${repoName}/${cleanPath}`;
  } else {
    // Local development or other hosting
    finalPath = `/${cleanPath}`;
  }
  
  console.log(`getPublicPath: ${path} -> ${finalPath}`);
  return finalPath;
} 