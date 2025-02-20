"use client";

import { useEffect, useState, useRef } from "react";
import { useInView } from "react-intersection-observer";
import "katex/dist/katex.min.css";
import { InlineMath, BlockMath } from "react-katex";
import Image from "next/image";
import Link from "next/link";
import * as d3 from "d3";

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
  showOriginal,
}: {
  ascii: string;
  shouldAddNoise: boolean;
  noiseLevel: number;
  showOriginal: boolean;
}) {
  const [noisyAscii, setNoisyAscii] = useState(ascii);

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

interface DataRow {
  raw: number;
  noise: number;
}

interface ChartRef extends HTMLDivElement {
  x: d3.ScaleBand<string>;
  y: d3.ScaleLinear<number, number>;
  innerHeight: number;
}

export default function Home() {
  const [monaLisa, setMonaLisa] = useState("");
  const [noiseLevel, setNoiseLevel] = useState(0);
  const [showOriginal, setShowOriginal] = useState(false);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const [selectedEpsilon, setSelectedEpsilon] = useState<number | null>(null);
  const { ref: addNoiseML, inView: pastNoiseTrigger } = useInView({
    threshold: 0.5,
    triggerOnce: true,
  });
  const { ref: titleRef, inView: isTitleVisible } = useInView({
    threshold: 0.5,
  });

  // Refs for sections
  const whatIsDPRef = useRef<HTMLElement>(null);
  const chartRef = useRef<ChartRef>(null);

  const scrollToSection = (ref: React.RefObject<HTMLElement | null>) => {
    if (ref.current) {
      ref.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  // Convert epsilon to probability (this is a simplified conversion for demonstration)
  const epsilonToProbability = (epsilon: number) => {
    return 1 / (1 + Math.exp(epsilon));
  };

  // Handle epsilon button click
  const handleEpsilonClick = (epsilon: number) => {
    setSelectedEpsilon(epsilon);
    const targetNoise = epsilonToProbability(epsilon);

    // Animate to new noise level
    let currentNoise = noiseLevel;
    const duration = 1000; // 1 second
    const steps = 20;
    const increment = (targetNoise - currentNoise) / steps;
    const stepDuration = duration / steps;

    const timer = setInterval(() => {
      if (Math.abs(currentNoise - targetNoise) > Math.abs(increment)) {
        currentNoise += increment;
        setNoiseLevel(currentNoise);
      } else {
        setNoiseLevel(targetNoise);
        clearInterval(timer);
      }
    }, stepDuration);
  };

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

  const [epsilon, setEpsilon] = useState(1);
  const [scale, setScale] = useState(1);
  const baseData = [
    { raw: 4, noise: 0 },
    { raw: 8, noise: 0 },
    { raw: 6, noise: 0 },
  ];
  const [data, setData] = useState<DataRow[]>(baseData);

  // Generate Laplace noise
  const generateLaplaceNoise = (scale: number) => {
    const u = Math.random() - 0.5;
    return -scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
  };

  // Update data when scale changes
  useEffect(() => {
    setData(
      baseData.map((row) => ({
        raw: row.raw * scale,
        noise: generateLaplaceNoise(1 / (epsilon / 3)),
      }))
    );
  }, [scale, epsilon]);

  // Get max value for y-scale
  const getYDomain = () => {
    const maxRaw = Math.max(...data.map((d) => d.raw));
    return maxRaw * 1.5; // Add 20% padding
  };

  // Update D3 visualization
  const updateChart = () => {
    if (!chartRef.current) return;

    // Clear previous chart
    d3.select(chartRef.current).selectAll("*").remove();

    // Chart dimensions
    const width = chartRef.current.clientWidth;
    const height = 400;
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create SVG
    const svg = d3
      .select(chartRef.current)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .style("overflow", "visible");

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`)
      .style("overflow", "visible");

    // Scales
    const x = d3
      .scaleBand()
      .domain(data.map((_, i) => `Bar ${i + 1}`))
      .range([0, innerWidth])
      .padding(0.3);

    const y = d3
      .scaleLinear()
      .domain([0, getYDomain()])
      .range([innerHeight, 0]);

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x));

    g.append("g").call(d3.axisLeft(y));

    // Create initial bars
    data.forEach((d, i) => {
      // Raw data bars
      g.append("rect")
        .attr("class", "bar-raw")
        .attr("x", x(`Bar ${i + 1}`) || 0)
        .attr("width", x.bandwidth())
        .attr("y", y(d.raw))
        .attr("height", innerHeight - y(d.raw))
        .attr("fill", "rgba(118, 171, 174, 0.5)");

      // Private data bars
      g.append("rect")
        .attr("class", "bar-private")
        .attr("x", x(`Bar ${i + 1}`) || 0)
        .attr("width", x.bandwidth())
        .attr("y", Math.min(y(0), y(d.raw + d.noise))) // Handle negative values
        .attr("height", Math.abs(y(d.raw + d.noise) - y(0))) // Correct height for both positive and negative values
        .attr("fill", "none")
        .attr("stroke", "#76ABAE")
        .attr("stroke-width", 2);
    });

    // Add legend
    const legend = svg
      .append("g")
      .attr(
        "transform",
        `translate(${width - margin.right - 100}, ${margin.top})`
      );

    legend
      .append("rect")
      .attr("x", 0)
      .attr("width", 15)
      .attr("height", 15)
      .attr("fill", "rgba(118, 171, 174, 0.5)");

    legend
      .append("rect")
      .attr("x", 0)
      .attr("y", 20)
      .attr("width", 15)
      .attr("height", 15)
      .attr("fill", "none")
      .attr("stroke", "#76ABAE")
      .attr("stroke-width", 2);

    legend
      .append("text")
      .attr("x", 20)
      .attr("y", 12)
      .text("Raw Data")
      .attr("class", "text-sm");

    legend
      .append("text")
      .attr("x", 20)
      .attr("y", 32)
      .text("Private Data")
      .attr("class", "text-sm");

    // Store scales for updates
    chartRef.current.x = x;
    chartRef.current.y = y;
    chartRef.current.innerHeight = innerHeight;
  };

  // Initial chart creation
  useEffect(() => {
    updateChart();
  }, []); // Empty dependency array means this only runs once

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      updateChart();
      // Update bars with current data after resize
      if (!chartRef.current) return;
      const { x, y, innerHeight } = chartRef.current;
      const g = d3.select(chartRef.current).select("svg g");
      g.selectAll(".bar-private")
        .data(data)
        .attr("x", (d, i) => x(`Bar ${i + 1}`) || 0)
        .attr("width", x.bandwidth())
        .attr("y", (d) => Math.min(y(0), y(d.raw + d.noise)))
        .attr("height", (d) => Math.abs(y(d.raw + d.noise) - y(0)));

      g.selectAll(".bar-raw")
        .data(data)
        .attr("x", (d, i) => x(`Bar ${i + 1}`) || 0)
        .attr("width", x.bandwidth())
        .attr("y", (d) => y(d.raw))
        .attr("height", (d) => innerHeight - y(d.raw));
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [data]);

  // Update bars when data changes
  useEffect(() => {
    if (!chartRef.current) return;
    updateChart();
    const { x, y } = chartRef.current;
    const g = d3.select(chartRef.current).select("svg g");

    g.selectAll(".bar-private")
      .data(data)
      .transition()
      .duration(750)
      .ease(d3.easeCubicOut)
      .attr("x", (d, i) => x(`Bar ${i + 1}`) || 0)
      .attr("width", x.bandwidth())
      .attr("y", (d) => Math.min(y(0), y(d.raw + d.noise)))
      .attr("height", (d) => Math.abs(y(d.raw + d.noise) - y(0)));
  }, [data]);

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
    <div className="h-screen overflow-y-scroll snap-y snap-mandatory bg-primary-white text-primary-black">
      {/* Navigation Bar */}
      <div
        className={`fixed top-0 left-0 right-0 z-50 bg-primary-white border-b border-primary-gray/10 transition-all duration-300 ${
          !isTitleVisible
            ? "translate-y-0 opacity-100"
            : "translate-y-[-100%] opacity-0"
        }`}
      >
        <nav className="flex justify-between items-center px-12 py-4">
          <h3 className="text-lg font-semibold">Privacy in Practice</h3>
          <ul className="flex gap-8">
            <li
              onClick={() => scrollToSection(whatIsDPRef)}
              className="text-accent hover:text-primary-gray transition-colors cursor-pointer"
            >
              1. What is Differential Privacy?
            </li>
            <li className="hover:text-accent transition-colors cursor-pointer">
              2. How We Applied DP
            </li>
            <li className="hover:text-accent transition-colors cursor-pointer">
              3. The Feasibility of Applying DP
            </li>
          </ul>
        </nav>
      </div>

      {/* Split view content */}
      <div className="flex">
        {/* Left side - content */}
        <div className="w-1/2">
          {/* Title Section */}
          <section
            ref={titleRef}
            className="h-screen snap-start flex flex-col justify-center p-12"
          >
            <h1 className="text-6xl font-bold mb-4">Privacy in Practice</h1>
            <h2 className="text-2xl font-semibold mb-2">
              The Feasibility of Differential Privacy for Telemetry Analysis
            </h2>
            <div className="mb-8">
              <p className="text-primary-gray">
                Tyler Kurpanek & Chris Lum & Bradley Nathanson & Trey Scheid
              </p>
              <p className="text-primary-gray">Mentor: Yu-Xiang Wang</p>
              <p className="text-primary-gray mt-4">
                <Link
                  href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-accent transition-colors cursor-pointer decoration-dotted underline"
                >
                  Report
                </Link>{" "}
                |{" "}
                <Link
                  href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-accent transition-colors cursor-pointer decoration-dotted underline"
                >
                  Poster
                </Link>{" "}
                |{" "}
                <Link
                  href="https://github.com/Trey-Scheid/privacy-in-practice"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-accent transition-colors cursor-pointer decoration-dotted underline"
                >
                  Github
                </Link>
              </p>
            </div>
            <div>
              <h2 className="text-xl font-semibold mb-2">Table of Contents</h2>
              <ul className="space-y-2">
                <li
                  onClick={() => scrollToSection(whatIsDPRef)}
                  className="hover:text-accent transition-colors cursor-pointer"
                >
                  1. What is Differential Privacy?
                </li>
                <li className="hover:text-accent transition-colors cursor-pointer">
                  2. How We Applied DP
                </li>
                <li className="hover:text-accent transition-colors cursor-pointer">
                  3. The Feasibility of Applying DP
                </li>
              </ul>
            </div>
            <div className="mt-36">
              <h2 className="text-xl font-semibold mb-2 text-accent">
                ↓ What is Differential Privacy? ↓
              </h2>
            </div>
          </section>

          {/* First Paragraph */}
          <section
            ref={whatIsDPRef}
            className="h-screen snap-start flex flex-col justify-center p-12"
          >
            <div className="prose prose-lg max-w-none">
              <h1 className="text-4xl font-bold mb-4">
                What is Differential Privacy?
              </h1>
              <p className="text-xl">
                In today&apos;s data-driven world, the need to protect
                individual privacy while maintaining the utility of data
                analysis has become increasingly crucial. Differential Privacy
                (DP) emerges as a mathematical framework that provides strong
                privacy guarantees while allowing meaningful statistical
                analysis
              </p>
            </div>
          </section>

          {/* Second Paragraph */}
          <section className="h-screen snap-start flex flex-col justify-center p-12">
            <div className="prose prose-lg max-w-none">
              <p className="text-xl">
                At its core, differential privacy ensures that the presence or
                absence of any individual&apos;s data in a dataset does not
                significantly affect the results of any analysis performed on
                that dataset. This is achieved by carefully introducing random
                noise into the computation process, making it virtually
                impossible to reverse-engineer individual records while
                preserving the overall statistical patterns in the data
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
                But what just happened? We added noise to the image of Mona Lisa
                by probabilistically flipping each pixel. This way, you can
                still see the big picture, but the individual pixels&apos; have
                some deniability as to what their original data was
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
                    className="w-full h-2 bg-primary-gray rounded-lg appearance-none cursor-pointer accent-accent"
                  />
                </div>
                <div className="w-1/2 text-lg font-medium text-primary-gray">
                  Probability: {noiseLevel.toFixed(2)}
                </div>
              </div>
              <div className="mt-4 flex">
                <button
                  onMouseDown={() => setShowOriginal(true)}
                  onMouseUp={() => setShowOriginal(false)}
                  onMouseLeave={() => setShowOriginal(false)}
                  className="px-6 py-2 bg-primary-gray text-primary-white rounded-lg hover:bg-accent transition-colors"
                >
                  Show Original
                </button>
              </div>
            </div>
          </section>

          {/* Fourth Paragraph */}
          <section className="h-screen snap-start flex flex-col justify-center p-12">
            <div className="prose prose-lg max-w-none">
              <p className="text-xl">
                What you see is a Differential Privacy technique called
                Randomized Response. It&apos;s a simple method that satisfies
                the{" "}
                <span className="font-bold text-accent">
                  Definition of Differential Privacy
                </span>
              </p>
            </div>
          </section>

          {/* Fifth Paragraph */}
          <section className="h-screen snap-start flex flex-col justify-center p-12">
            <div className="prose prose-lg max-w-none">
              <p className="text-xl">
                <span className="font-bold text-accent">
                  Definition of Differential Privacy
                </span>{" "}
                states that for any two datasets that differ in exactly one
                record, the probability of getting the same output from an
                private algorithm should be similar. This paves the way for a{" "}
                <span
                  className="text-accent hover:text-primary-gray cursor-pointer transition-colors underline decoration-dotted"
                  onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                >
                  mathematical measure of privacy
                </span>{" "}
                and a way to quantify the privacy of an algorithm.
              </p>
              <div
                className={`mt-8 transition-all duration-500 overflow-hidden ${
                  showTechnicalDetails
                    ? "max-h-[500px] opacity-100"
                    : "max-h-0 opacity-0"
                }`}
              >
                <div className="bg-primary-gray/5 p-6 rounded-lg border border-primary-gray/10">
                  <h3 className="text-lg font-semibold mb-4">
                    <InlineMath math="\varepsilon\text{-}\delta" /> Differential
                    Privacy
                  </h3>
                  <p className="text-lg mb-4">
                    A randomized algorithm <InlineMath math="\mathcal{M}" />{" "}
                    satisfies <InlineMath math="(\varepsilon,\delta)" />
                    -differential privacy if for all neighboring datasets{" "}
                    <InlineMath math="\mathcal{D}" /> and{" "}
                    <InlineMath math="\mathcal{D}'" /> and for all possible
                    outputs <InlineMath math="\mathcal{S}" />:
                  </p>
                  <div className="bg-primary-gray/10 p-2 rounded-lg my-6 flex justify-center">
                    <BlockMath math="P(\mathcal{M}(\mathcal{D}) \in \mathcal{S}) \leq e^\varepsilon \cdot P(\mathcal{M}(\mathcal{D}') \in \mathcal{S}) + \delta" />
                  </div>
                  <p className="text-lg">Where:</p>
                  <ul className="list-disc ml-6 mt-2 space-y-2">
                    <li>
                      <InlineMath math="\varepsilon" /> (epsilon) controls the
                      privacy budget - smaller values mean stronger privacy and
                      more noise
                    </li>
                    <li>
                      <InlineMath math="\delta" /> (delta) is the probability of
                      the privacy guarantee failing
                    </li>
                  </ul>
                  <div className="mt-4 space-y-4">
                    <div className="flex items-center gap-2">
                      <span>Try changing epsilon!</span>
                      <div className="flex-1 flex justify-between items-center mx-4">
                        {[0.5, 1, 2, 10].map((epsilon) => (
                          <span
                            key={epsilon}
                            onClick={() => handleEpsilonClick(epsilon)}
                            className={`cursor-pointer ${
                              selectedEpsilon === epsilon
                                ? "text-accent font-bold"
                                : "text-primary-gray hover:text-accent"
                            } transition-colors`}
                          >
                            <InlineMath math={`\\varepsilon=${epsilon}`} />
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                  <p className="mt-4">
                    Read more:{" "}
                    <Link
                      href="https://en.wikipedia.org/wiki/Differential_privacy"
                      target="_blank"
                      className="text-accent hover:text-primary-gray cursor-pointer transition-colors underline decoration-dotted"
                    >
                      Wikipedia
                    </Link>
                  </p>
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* Right side - Mona Lisa */}
        <div className="w-1/2 sticky top-0 h-screen bg-primary-white flex items-center justify-center overflow-hidden">
          <div className="transform scale-125">
            <MonaLisa
              ascii={monaLisa}
              shouldAddNoise={pastNoiseTrigger}
              noiseLevel={noiseLevel}
              showOriginal={showOriginal}
            />
          </div>
        </div>
      </div>

      {/* Centered content section */}
      <div className="bg-primary-white">
        {/* Sixth Paragraph - Full Width Centered */}
        <section className="h-screen snap-start flex flex-col justify-center items-center p-12">
          <div className="prose prose-lg max-w-3xl mx-auto text-center">
            <p className="text-xl">
              What&apos;s important to note is that Differential Privacy is a
              property of an algorithm, not a property of the data. Another
              intuitive{" "}
              <Link
                href="https://en.wikipedia.org/wiki/Privacy-enhancing_technologies"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-accent cursor-pointer transition-colors underline decoration-dotted"
              >
                Privacy-Enhancing Technology
              </Link>{" "}
              might be to anonymize data by removing personally identifiable
              information, but this isn&apos;t enough to guarantee privacy on
              its own.
            </p>
          </div>
        </section>

        {/* Seventh Paragraph - Full Width Centered */}
        <section className="h-screen snap-start flex justify-between items-center p-12 mx-12">
          <div className="w-1/2 prose prose-lg max-w-none">
            <p className="text-xl">
              In 2006, Netflix created the{" "}
              <Link
                href="https://en.wikipedia.org/wiki/Netflix_Prize"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-accent cursor-pointer transition-colors underline decoration-dotted"
              >
                Netflix Prize
              </Link>
              , a competition to improve Netflix&apos;s movie recommendation
              algorithm. The dataset used in the competition contained 100
              million ratings from 480,000 users on 17,770 movies, anonymized by
              removing personally identifiable information. One year later,
              using iMDB ratings as a reference, two researchers from UT Austin
              were able to{" "}
              <Link
                href="https://arxiv.org/abs/cs/0610105"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-accent cursor-pointer transition-colors underline decoration-dotted"
              >
                deanonymize the 99% of the users in the dataset
              </Link>
              . This is why the rigor of DP is so important.
            </p>
          </div>
          <div className="w-1/2 flex justify-center items-center">
            <div className="w-full h-full flex justify-center items-center">
              <div className="relative w-[70%]">
                <Image
                  src="/netflix.svg"
                  alt="Netflix Logo"
                  className="w-full h-auto"
                  width={0}
                  height={0}
                  sizes="(max-width: 768px) 100vw, 50vw"
                  style={{ width: "100%", height: "auto" }}
                  priority
                />
              </div>
            </div>
          </div>
        </section>

        {/* Eighth Paragraph - Interactive Visualization */}
        <section className="h-screen snap-start flex flex-col justify-center p-12">
          <div className="prose prose-lg max-w-3xl mx-auto mb-12 text-center">
            <p className="text-xl">
              Another simple way to privatize data is to add noise to each of
              the values you plan to release. Here, we&apos;re releasing the raw
              counts of each category by adding noise to each of the counts.
            </p>
          </div>
          <div className="flex justify-between items-start gap-8">
            {/* Left side - Table */}
            <div className="w-1/2">
              <table className="w-full border-collapse">
                <thead>
                  <tr>
                    <th className="p-4 text-left border-b border-primary-gray/10">
                      <div className="flex items-center gap-2">
                        <div>Raw Data</div>
                        <div className="relative group">
                          <div className="w-5 h-5 rounded-full bg-primary-gray/10 flex items-center justify-center text-primary-gray cursor-hel">
                            ?
                          </div>
                          <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-64 p-3 bg-primary-gray text-primary-white rounded-lg text-sm opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-10">
                            Notice how having more data means that the noise
                            affects the big picture less.
                          </div>
                        </div>
                      </div>
                      <div className="mt-2">
                        <input
                          type="range"
                          min="1"
                          max="10"
                          step="0.5"
                          value={scale}
                          onChange={(e) => setScale(parseFloat(e.target.value))}
                          className="w-full h-2 bg-primary-gray rounded-lg appearance-none cursor-pointer accent-accent"
                        />
                        <div className="text-sm text-primary-gray mt-1">
                          Scale = {scale.toFixed(1)}x
                        </div>
                      </div>
                    </th>
                    <th className="p-4 text-left border-b border-primary-gray/10">
                      <div className="flex items-center gap-2">
                        <div>Introduce Noise</div>
                        <div className="relative group">
                          <div className="w-5 h-5 rounded-full bg-primary-gray/10 flex items-center justify-center text-primary-gray cursor-help">
                            ?
                          </div>
                          <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-64 p-3 bg-primary-gray text-primary-white rounded-lg text-sm opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-10">
                            The noise follows a Laplace distribution with scale
                            proportional to 1/ε. This mechanism satisfies (ε,
                            0)-differential privacy, meaning that the
                            probability of the privacy guarantee failing is 0.
                          </div>
                        </div>
                      </div>
                      <div className="mt-2">
                        <input
                          type="range"
                          min="0.1"
                          max="3"
                          step="0.1"
                          value={epsilon}
                          onChange={(e) =>
                            setEpsilon(parseFloat(e.target.value))
                          }
                          className="w-full h-2 bg-primary-gray rounded-lg appearance-none cursor-pointer accent-accent"
                        />
                        <div className="text-sm text-primary-gray mt-1">
                          ε = {epsilon.toFixed(1)}
                        </div>
                      </div>
                    </th>
                    <th className="p-4 text-left border-b border-primary-gray/10">
                      Privatized Data
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {data.map((row, i) => (
                    <tr key={i}>
                      <td className="p-4 border-b border-primary-gray/10">
                        {row.raw}
                      </td>
                      <td className="p-4 border-b border-primary-gray/10">
                        {row.noise.toFixed(2)}
                      </td>
                      <td className="p-4 border-b border-primary-gray/10">
                        {(row.raw + row.noise).toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {/* Right side - Chart */}
            <div className="w-1/2" ref={chartRef} />
          </div>
        </section>
      </div>
    </div>
  );
}
