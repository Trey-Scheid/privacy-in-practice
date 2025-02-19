'use client';

import { useEffect, useState } from 'react';
import { useInView } from 'react-intersection-observer';

function addNoiseToAscii(ascii: string, noiseLevel: number = 0.3) {
  return ascii.split('').map(char => {
    // Only flip if it's one of our ASCII characters
    if ((char === ' ' || char === '@') && Math.random() < noiseLevel) {
      // Flip the character
      return char === ' ' ? '@' : ' ';
    }
    return char;
  }).join('');
}

function MonaLisa({ ascii, shouldAddNoise }: { ascii: string, shouldAddNoise: boolean }) {
  const [noisyAscii, setNoisyAscii] = useState(ascii);
  
  useEffect(() => {
    if (shouldAddNoise) {
      setNoisyAscii(addNoiseToAscii(ascii));
    } else {
      setNoisyAscii(ascii);
    }
  }, [shouldAddNoise, ascii]);

  return (
    <pre className="font-mono text-[0.6rem] leading-[0.6rem] whitespace-pre text-gray-800">
      {noisyAscii}
    </pre>
  );
}

export default function Home() {
  const [monaLisa, setMonaLisa] = useState('');
  const { ref: secondParaRef, inView: pastSecondPara } = useInView({
    threshold: 0.5,
    triggerOnce: false
  });

  useEffect(() => {
    fetch('/monalisa.txt')
      .then(response => response.text())
      .then(content => setMonaLisa(content))
      .catch(error => {
        console.error('Failed to load Mona Lisa:', error);
        setMonaLisa('');
      });
  }, []);

  return (
    <div className="min-h-screen flex">
      {/* Left side - scrollable content */}
      <div className="w-1/2 p-12 overflow-y-auto">
        <div className="h-full">
          <h1 className="text-4xl font-bold mb-4">Novel Techniques in Private Data Analysis</h1>

          <div className="mb-8">
            <h2 className="text-xl font-semibold mb-2">Authors</h2>
            <ul className="space-y-1 text-gray-600">
              <p className="text-gray-600">Chris | Bradley | Trey | Tyler </p>
            </ul>
          </div>

          <div>
            <h2 className="text-xl font-semibold mb-2">Table of Contents</h2>
            <ul className="space-y-2">
              <li className="hover:text-blue-500 cursor-pointer">1. Introduction to Differential Privacy</li>
              <li className="hover:text-blue-500 cursor-pointer">2. Novel Techniques</li>
              <li className="hover:text-blue-500 cursor-pointer">3. Implementation and Results</li>
            </ul>
          </div>
        </div>

        <h1 className="text-4xl font-bold mb-8">What is Differential Privacy?</h1>
        
        <div className="prose prose-lg max-w-none">
          <p className="mb-6">
            In today's data-driven world, the need to protect individual privacy while maintaining the utility of data analysis has become increasingly crucial. Differential Privacy (DP) emerges as a mathematical framework that provides strong privacy guarantees while allowing meaningful statistical analysis.
          </p>

          <p ref={secondParaRef} className="mb-6">
            At its core, differential privacy ensures that the presence or absence of any individual's data in a dataset does not significantly affect the results of any analysis performed on that dataset. This is achieved by carefully introducing random noise into the computation process, making it virtually impossible to reverse-engineer individual records while preserving the overall statistical patterns in the data.
          </p>

          <p className="mb-6">
            The concept can be illustrated through a simple analogy: imagine trying to determine whether someone participated in a survey by looking at the survey's results. With differential privacy, the results would be nearly identical regardless of whether any particular individual participated or not, effectively masking their participation while maintaining the survey's overall accuracy.
          </p>
        </div>
      </div>

      {/* Right side - fixed Mona Lisa */}
      <div className="w-1/2 fixed right-0 top-0 h-screen bg-white flex items-center justify-center overflow-hidden">
        <div className="transform scale-110">
          <MonaLisa ascii={monaLisa} shouldAddNoise={pastSecondPara} />
        </div>
      </div>
    </div>
  );
}
