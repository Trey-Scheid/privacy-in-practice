"use client";

import { useEffect, useState } from "react";
import { useInView } from "react-intersection-observer";

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

function MonaLisa({
  ascii,
  shouldAddNoise,
  noiseLevel,
}: {
  ascii: string;
  shouldAddNoise: boolean;
  noiseLevel: number;
}) {
  const [noisyAscii, setNoisyAscii] = useState(ascii);

  useEffect(() => {
    if (shouldAddNoise) {
      setNoisyAscii(addNoiseToAscii(ascii, noiseLevel));
    } else {
      setNoisyAscii(ascii);
    }
  }, [shouldAddNoise, ascii, noiseLevel]);

  return (
    <pre className="font-mono text-[0.6rem] leading-[0.6rem] whitespace-pre text-gray-800">
      {noisyAscii}
    </pre>
  );
}

export default function Home() {
  const [monaLisa, setMonaLisa] = useState("");
  const [noiseLevel, setNoiseLevel] = useState(0);
  const { ref: addNoiseML, inView: pastNoiseTrigger } = useInView({
    threshold: 0.5,
    triggerOnce: true,
  });

  // Animate noise level when section comes into view
  useEffect(() => {
    if (pastNoiseTrigger) {
      let currentNoise = 0;
      const targetNoise = 0.2;
      const duration = 1500; // 1.5 seconds
      const steps = 20; // Number of steps in the animation
      const increment = targetNoise / steps;
      const stepDuration = duration / steps;

      const timer = setInterval(() => {
        if (currentNoise < targetNoise) {
          currentNoise += increment;
          setNoiseLevel(currentNoise);
        } else {
          clearInterval(timer);
        }
      }, stepDuration);

      return () => clearInterval(timer);
    } else {
      setNoiseLevel(0);
    }
  }, [pastNoiseTrigger]);

  // Handle manual slider changes
  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setNoiseLevel(parseFloat(e.target.value));
  };

  useEffect(() => {
    fetch("/monalisa.txt")
      .then((response) => response.text())
      .then((content) => setMonaLisa(content))
      .catch((error) => {
        console.error("Failed to load Mona Lisa:", error);
        setMonaLisa("");
      });
  }, []);

  return (
    <div className="h-screen overflow-y-scroll snap-y snap-mandatory">
      <div className="relative">
        {/* Content container */}
        <div className="flex">
          {/* Left side - content */}
          <div className="w-1/2">
            {/* Title Section */}
            <section className="h-screen snap-start flex flex-col justify-center p-12">
              <h1 className="text-4xl font-bold mb-4">Privacy in Practice</h1>
              <h2 className="text-2xl font-semibold mb-2">
                The Feasibility of Differential Privacy for Telemetry Analysis
              </h2>
              <div className="mb-8">
                <p className="text-gray-600">
                  Tyler Kurpanek & Chris Lum & Bradley Nathanson & Trey Scheid
                </p>
                <p className="text-gray-600">Mentor: Yu-Xiang Wang</p>
              </div>
              <div>
                <h2 className="text-xl font-semibold mb-2">
                  Table of Contents
                </h2>
                <ul className="space-y-2">
                  <li className="hover:text-blue-500 cursor-pointer">
                    1. What is Differential Privacy?
                  </li>
                  <li className="hover:text-blue-500 cursor-pointer">
                    2. How We Applied DP
                  </li>
                  <li className="hover:text-blue-500 cursor-pointer">
                    3. The Feasibility of Applying DP
                  </li>
                </ul>
              </div>
              <div className="mt-36">
                <h2 className="text-xl font-semibold mb-2">
                  ↓ What is Differential Privacy? ↓
                </h2>
              </div>
            </section>

            {/* First Paragraph */}
            <section className="h-screen snap-start flex flex-col justify-center p-12">
              <div className="prose prose-lg max-w-none">
                <h1 className="text-4xl font-bold mb-4">
                  What is Differential Privacy?
                </h1>
                <p className="text-xl">
                  In today's data-driven world, the need to protect individual
                  privacy while maintaining the utility of data analysis has
                  become increasingly crucial. Differential Privacy (DP) emerges
                  as a mathematical framework that provides strong privacy
                  guarantees while allowing meaningful statistical analysis.
                </p>
              </div>
            </section>

            {/* Second Paragraph */}
            <section className="h-screen snap-start flex flex-col justify-center p-12">
              <div className="prose prose-lg max-w-none">
                <p className="text-xl">
                  At its core, differential privacy ensures that the presence or
                  absence of any individual's data in a dataset does not
                  significantly affect the results of any analysis performed on
                  that dataset. This is achieved by carefully introducing random
                  noise into the computation process, making it virtually
                  impossible to reverse-engineer individual records while
                  preserving the overall statistical patterns in the data.
                </p>
              </div>
            </section>

            {/* Third Paragraph */}
            <section
              ref={addNoiseML}
              className="h-screen snap-start flex flex-col justify-center p-12"
            >
              <div className="prose prose-lg max-w-none">
                <p className="text-xl">...while maintaining the big picture!</p>
                <p className="text-xl mt-6">
                  But what just happened? We added noise to the image of Mona
                  Lisa by probabilistically flipping each pixel. This way, you
                  can still see the big picture, but the individual pixels&apos;
                  data is now randomized.
                </p>
                <p className="text-xl mt-6">Try for yourself!</p>
                <div className="mt-8 flex items-center gap-4">
                  <div className="w-1/2">
                    <input
                      type="range"
                      id="noise-level"
                      min="0"
                      max="0.5"
                      step="0.01"
                      value={noiseLevel}
                      onChange={handleSliderChange}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                  <div className="w-1/2 text-lg font-medium text-gray-700">
                    Probability: {noiseLevel.toFixed(2)}
                  </div>
                </div>
              </div>
            </section>

            {/* Fourth Paragraph */}
            <section className="h-screen snap-start flex flex-col justify-center p-12">
              <div className="prose prose-lg max-w-none">
                <p className="text-xl">
                  At its core, differential privacy ensures that the presence or
                  absence of any individual's data in a dataset does not
                  significantly affect the results of any analysis performed on
                  that dataset. This is achieved by carefully introducing random
                  noise into the computation process, making it virtually
                  impossible to reverse-engineer individual records while
                  preserving the overall statistical patterns in the data.
                </p>
              </div>
            </section>
          </div>

          {/* Right side - static width holder */}
          <div className="w-1/2" />
        </div>

        {/* Overlay Mona Lisa */}
        <div className="w-1/2 fixed right-0 top-0 h-screen bg-white flex items-center justify-center overflow-hidden pointer-events-none">
          <div className="transform scale-110">
            <MonaLisa
              ascii={monaLisa}
              shouldAddNoise={pastNoiseTrigger}
              noiseLevel={noiseLevel}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
