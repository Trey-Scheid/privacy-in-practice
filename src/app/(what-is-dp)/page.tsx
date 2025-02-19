import { promises as fs } from 'fs';
import path from 'path';

async function getMonaLisa() {
  const filePath = path.join(process.cwd(), 'public', 'monalisa.txt');
  const content = await fs.readFile(filePath, 'utf8');
  return content;
}

export default async function Home() {
  const monaLisa = await getMonaLisa();

  return (
    <div className="min-h-screen flex">
      {/* Left side - scrollable content */}
      <div className="w-1/2 p-12 overflow-y-auto">
        <h1 className="text-4xl font-bold mb-4">Novel Techniques in Private Data Analysis</h1>
        
        <div className="mb-8">
          <h2 className="text-xl font-semibold mb-2">Authors</h2>
          <ul className="space-y-1 text-gray-600">
            <li>Author One</li>
            <li>Author Two</li>
            <li>Author Three</li>
          </ul>
        </div>

        <div>
          <h2 className="text-xl font-semibold mb-2">Table of Contents</h2>
          <ul className="space-y-2">
            <li className="hover:text-blue-500 cursor-pointer">1. Introduction to Differential Privacy</li>
            <li className="hover:text-blue-500 cursor-pointer">2. Novel Techniques</li>
            <li className="hover:text-blue-500 cursor-pointer">3. Implementation and Results</li>
            <li className="hover:text-blue-500 cursor-pointer">4. Future Directions</li>
          </ul>
        </div>
      </div>

      {/* Right side - fixed Mona Lisa */}
      <div className="w-1/2 fixed right-0 top-0 h-screen bg-gray-900 text-green-500 flex items-center justify-center overflow-hidden">
        <pre className="font-mono text-[0.5rem] leading-[0.5rem] whitespace-pre">
          {monaLisa}
        </pre>
      </div>
    </div>
  );
}
