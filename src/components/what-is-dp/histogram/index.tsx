import Link from "next/link";
import Image from "next/image";
import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

interface DataRow {
  raw: number;
  noise: number;
}

interface ChartRef extends HTMLDivElement {
  x: d3.ScaleBand<string>;
  y: d3.ScaleLinear<number, number>;
  innerHeight: number;
}

export function HistogramSection() {
  const chartRef = useRef<ChartRef>(null);
  const [epsilon, setEpsilon] = useState(1);
  const [scale, setScale] = useState(1);
  const scaleValues = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16];
  const baseData = [
    { raw: 16, noise: 0 },
    { raw: 32, noise: 0 },
    { raw: 24, noise: 0 },
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
        .attr("y", Math.min(y(0), y(d.raw + d.noise)))
        .attr("height", Math.abs(y(d.raw + d.noise) - y(0)))
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

  return (
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
            information, but this isn&apos;t enough to guarantee privacy on its
            own.
          </p>
        </div>
      </section>

      {/* Netflix Prize Section */}
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
            algorithm. The dataset used in the competition contained 100 million
            ratings from 480,000 users on 17,770 movies, anonymized by removing
            personally identifiable information. One year later, using iMDB
            ratings as a reference, two researchers from UT Austin were able to{" "}
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

      {/* Interactive Visualization Section */}
      <section className="h-screen snap-start flex flex-col justify-center p-12">
        <div className="prose prose-lg max-w-3xl mx-auto mb-12 text-center">
          <p className="text-xl">
            Another simple way to privatize data is to add noise to each of the
            values you plan to release. Here, we&apos;re releasing the raw
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
                        min="0"
                        max="8"
                        step="1"
                        value={scaleValues.indexOf(scale)}
                        onChange={(e) =>
                          setScale(scaleValues[parseInt(e.target.value)])
                        }
                        className="w-full h-2 bg-primary-gray rounded-lg appearance-none cursor-pointer accent-accent"
                      />
                      <div className="text-sm text-primary-gray mt-1">
                        Scale ={" "}
                        {scale < 1 ? `1/${Math.round(1 / scale)}` : `${scale}`}x
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
                          0)-differential privacy, meaning that the probability
                          of the privacy guarantee failing is 0.
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
                        onChange={(e) => setEpsilon(parseFloat(e.target.value))}
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
  );
}
