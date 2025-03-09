import { useEffect, useState } from "react";

function addNoiseToAscii(ascii: string, noiseLevel: number = 0.2) {
  return ascii
    .split("")
    .map((char) => {
      // Only flip if it's one of our ASCII characters
      if ((char === "." || char === "@") && Math.random() < noiseLevel) {
        // Flip the character
        return char === "." ? "@" : ".";
      }
      return char;
    })
    .join("");
}

interface MonaLisaProps {
  ascii: string;
  shouldAddNoise: boolean;
  noiseLevel: number;
  showOriginal: boolean;
}

export function MonaLisa({
  ascii,
  shouldAddNoise,
  noiseLevel,
  showOriginal,
}: MonaLisaProps) {
  const [noisyAscii, setNoisyAscii] = useState(ascii);

  // Update noise ONLY when props change
  useEffect(() => {
    if (shouldAddNoise && !showOriginal) {
      setNoisyAscii(addNoiseToAscii(ascii, noiseLevel));
    } else {
      setNoisyAscii(ascii);
    }
  }, [shouldAddNoise, ascii, noiseLevel, showOriginal]);

  return (
    <pre className="font-mono text-[0.75rem] leading-[0.75rem] whitespace-pre text-primary-black">
      {noisyAscii}
    </pre>
  );
}
