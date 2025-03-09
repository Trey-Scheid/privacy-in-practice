import { useEffect, useRef, useCallback } from "react";
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

interface ChartProps {
  data: DataRow[];
}

export function Chart({ data }: ChartProps) {
  const chartRef = useRef<ChartRef>(null);

  // Get max value for y-scale
  const getYDomain = useCallback(() => {
    const maxRaw = Math.max(...data.map((d) => d.raw));
    return maxRaw * 1.5; // Add 20% padding
  }, [data]);

  // Update D3 visualization
  const updateChart = useCallback(() => {
    if (!chartRef.current) return;

    // Clear previous chart
    d3.select(chartRef.current).selectAll("*").remove();

    // Chart dimensions
    const width = chartRef.current.clientWidth;
    const height = window.innerWidth < 768 ? 300 : 400; // Smaller height on mobile
    const margin = {
      top: 20,
      right: window.innerWidth < 768 ? 10 : 20,
      bottom: 30,
      left: window.innerWidth < 768 ? 30 : 40,
    };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create SVG
    const svg = d3
      .select(chartRef.current)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", `0 0 ${width} ${height}`) // Add viewBox for responsiveness
      .attr("preserveAspectRatio", "xMidYMid meet") // Preserve aspect ratio
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
  }, [data, getYDomain]);

  // Initial chart creation and handle window resize
  useEffect(() => {
    updateChart();

    const handleResize = () => {
      updateChart();
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [updateChart]);

  return <div className="w-full" ref={chartRef} />;
}
