import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

interface ChartRef extends HTMLDivElement {
  x: d3.ScaleBand<string>;
  y: d3.ScaleLinear<number, number>;
  svg?: d3.Selection<SVGSVGElement, unknown, null, undefined>;
}

interface ConfidenceData {
  label: string;
  nonPrivate: number;
  private: number;
}

const data: { [key: string]: ConfidenceData[] } = {
  "Image 1": [
    { label: "True", nonPrivate: 0.42, private: 0.48 },
    { label: "False", nonPrivate: 0.58, private: 0.52 },
  ],
  "Image 2": [
    { label: "True", nonPrivate: 0.98, private: 0.61 },
    { label: "False", nonPrivate: 0.02, private: 0.39 },
  ],
  "Image 3": [
    { label: "True", nonPrivate: 0.62, private: 0.58 },
    { label: "False", nonPrivate: 0.38, private: 0.42 },
  ],
};

const COLORS = {
  private: "#76ABAE",
  nonPrivate: "#2C3333",
};

export function ConfidenceChart() {
  const chartRef = useRef<ChartRef>(null);
  const [isPrivate, setIsPrivate] = useState(true);
  const [selectedDatum, setSelectedDatum] = useState("Image 1");

  // Initialize chart
  useEffect(() => {
    if (!chartRef.current || chartRef.current.svg) return;

    const width = chartRef.current.clientWidth;
    const height = window.innerWidth < 768 ? 250 : 300;
    const margin = {
      top: 20,
      right: window.innerWidth < 768 ? 10 : 20,
      bottom: 30,
      left: window.innerWidth < 768 ? 35 : 60,
    };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create SVG
    const svg = d3
      .select(chartRef.current)
      .append("svg")
      .attr("width", "100%")
      .attr("height", height)
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet");

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Scales
    const x = d3
      .scaleBand()
      .domain(["True", "False"])
      .range([0, innerWidth])
      .padding(0.3);

    const y = d3.scaleLinear().domain([0, 1]).range([innerHeight, 0]);

    // Add y-axis label
    svg
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", margin.left - 40)
      .attr("x", -(height / 2))
      .attr("text-anchor", "middle")
      .attr("fill", "#2C3333")
      .attr("font-size", "14px")
      .text("Model Confidence");

    // Add horizontal reference lines
    const lines = g.append("g").attr("class", "reference-lines");

    // 0% line (bottom axis)
    lines
      .append("line")
      .attr("x1", 0)
      .attr("x2", innerWidth)
      .attr("y1", innerHeight)
      .attr("y2", innerHeight)
      .attr("stroke", "#2C3333")
      .attr("stroke-width", 2);

    // 50% line
    lines
      .append("line")
      .attr("x1", 0)
      .attr("x2", innerWidth)
      .attr("y1", y(0.5))
      .attr("y2", y(0.5))
      .attr("stroke", "#2C3333")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "4,4")
      .attr("opacity", 0.3);

    // 100% line
    lines
      .append("line")
      .attr("x1", 0)
      .attr("x2", innerWidth)
      .attr("y1", y(1))
      .attr("y2", y(1))
      .attr("stroke", "#2C3333")
      .attr("stroke-width", 2);

    // Add percentage labels
    const labels = g.append("g").attr("class", "percentage-labels");

    [0, 0.5, 1].forEach((value) => {
      labels
        .append("text")
        .attr("x", -5)
        .attr("y", y(value))
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "middle")
        .attr("fill", "#2C3333")
        .attr("font-size", "12px")
        .text(d3.format(".0%")(value));
    });

    // Add x-axis labels
    g.append("g")
      .attr("class", "x-labels")
      .selectAll("text")
      .data(["True", "False"])
      .enter()
      .append("text")
      .attr("x", (d) => (x(d) || 0) + x.bandwidth() / 2)
      .attr("y", innerHeight + 20)
      .attr("text-anchor", "middle")
      .text((d) => d);

    // Create bars group
    g.append("g").attr("class", "bars");

    // Create labels group
    g.append("g").attr("class", "value-labels");

    // Store references
    chartRef.current.svg = svg;
    chartRef.current.x = x;
    chartRef.current.y = y;
  }, []);

  // Update chart data
  useEffect(() => {
    if (!chartRef.current?.svg || !chartRef.current.x || !chartRef.current.y)
      return;

    const svg = chartRef.current.svg;
    const x = chartRef.current.x;
    const y = chartRef.current.y;

    const g = svg.select("g");

    // Update bars
    const bars = g.select(".bars").selectAll("rect").data(data[selectedDatum]);

    // Enter new bars
    bars
      .enter()
      .append("rect")
      .attr("x", (d) => x(d.label) || 0)
      .attr("width", x.bandwidth())
      .attr("rx", 4)
      .attr("ry", 4)
      .merge(bars as any)
      .attr("fill", isPrivate ? COLORS.private : COLORS.nonPrivate)
      .transition()
      .duration(750)
      .ease(d3.easeCubicOut)
      .attr("y", (d) => y(isPrivate ? d.private : d.nonPrivate))
      .attr("height", (d) => y(0) - y(isPrivate ? d.private : d.nonPrivate));

    // Update value labels
    const labels = g
      .select(".value-labels")
      .selectAll("text")
      .data(data[selectedDatum]);

    // Remove old labels
    labels.exit().remove();

    // Enter new labels
    const labelEnter = labels
      .enter()
      .append("text")
      .attr("x", (d) => (x(d.label) || 0) + x.bandwidth() / 2)
      .attr("text-anchor", "middle")
      .style("font-weight", "500");

    // Update + Enter labels
    labelEnter
      .merge(labels as any)
      .raise() // Ensure labels are always on top
      .transition()
      .duration(750)
      .ease(d3.easeCubicOut)
      .attr("y", (d) => {
        const value = isPrivate ? d.private : d.nonPrivate;
        if (value <= 0.05) {
          // Small values: position above bar
          return y(value) - 10;
        } else {
          // Normal values: position inside bar
          return y(value) + (y(0) - y(value)) / 2 + 5;
        }
      })
      .attr("fill", (d) => {
        const value = isPrivate ? d.private : d.nonPrivate;
        // Use white text for values > 5% and inside bar
        return value > 0.05 ? "white" : "#2C3333";
      })
      .tween("text", function (d) {
        /* eslint-disable-next-line @typescript-eslint/no-this-alias */
        const node = this;
        const currentValue = parseFloat(node.textContent || "0") / 100;
        const targetValue = isPrivate ? d.private : d.nonPrivate;
        const i = d3.interpolate(currentValue, targetValue);
        return function (t) {
          node.textContent = d3.format(".0%")(i(t));
        };
      });
  }, [isPrivate, selectedDatum]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (!chartRef.current?.svg) return;

      const width = chartRef.current.clientWidth;
      const height = window.innerWidth < 768 ? 250 : 300;
      const margin = {
        top: 20,
        right: window.innerWidth < 768 ? 10 : 20,
        bottom: 30,
        left: window.innerWidth < 768 ? 35 : 60,
      };
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;

      // Update SVG viewBox
      chartRef.current.svg.attr("viewBox", `0 0 ${width} ${height}`);

      // Update scales
      chartRef.current.x.range([0, innerWidth]);
      chartRef.current.y.range([innerHeight, 0]);

      const g = chartRef.current.svg.select("g");

      // Update reference lines
      g.select(".reference-lines")
        .selectAll("line")
        .attr("x2", innerWidth)
        .each(function (_, i) {
          const line = d3.select(this);
          if (i === 0) {
            // 0% line
            line.attr("y1", innerHeight).attr("y2", innerHeight);
          } else if (i === 1) {
            // 50% line
            line
              .attr("y1", chartRef.current!.y(0.5))
              .attr("y2", chartRef.current!.y(0.5));
          } else {
            // 100% line
            line
              .attr("y1", chartRef.current!.y(1))
              .attr("y2", chartRef.current!.y(1));
          }
        });

      // Update percentage labels
      g.select(".percentage-labels")
        .selectAll("text")
        .each(function (_, i) {
          const value = [0, 0.5, 1][i];
          d3.select(this).attr("y", chartRef.current!.y(value));
        });

      // Update x-axis labels
      g.select(".x-labels")
        .selectAll("text")
        .attr(
          "x",
          (d) =>
            (chartRef.current!.x(d as string) || 0) +
            chartRef.current!.x.bandwidth() / 2
        )
        .attr("y", innerHeight + 20);

      // Update bars
      g.select(".bars")
        .selectAll("rect")
        .attr("x", (d: any) => chartRef.current!.x(d.label) || 0)
        .attr("width", chartRef.current!.x.bandwidth())
        .attr("y", (d: any) =>
          chartRef.current!.y(isPrivate ? d.private : d.nonPrivate)
        )
        .attr(
          "height",
          (d: any) =>
            chartRef.current!.y(0) -
            chartRef.current!.y(isPrivate ? d.private : d.nonPrivate)
        );

      // Update value labels
      g.select(".value-labels")
        .selectAll("text")
        .attr(
          "x",
          (d: any) =>
            (chartRef.current!.x(d.label) || 0) +
            chartRef.current!.x.bandwidth() / 2
        )
        .attr("y", (d: any) => {
          const value = isPrivate ? d.private : d.nonPrivate;
          if (value <= 0.05) {
            return chartRef.current!.y(value) - 10;
          } else {
            return (
              chartRef.current!.y(value) +
              (chartRef.current!.y(0) - chartRef.current!.y(value)) / 2 +
              5
            );
          }
        });

      // Update y-axis label position
      chartRef.current.svg
        .select("text")
        .attr("y", margin.left - 35)
        .attr("x", -(height / 2));
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [isPrivate]);

  return (
    <div className="flex flex-col items-center gap-6 md:gap-8 pt-4 md:pt-8">
      {/* Datum Selection */}
      <div className="flex gap-2 md:gap-4 items-center">
        {["Image 1", "Image 2", "Image 3"].map((datum) => (
          <button
            key={datum}
            onClick={() => setSelectedDatum(datum)}
            className={`px-3 md:px-4 py-2 rounded-lg transition-colors text-sm md:text-base ${
              selectedDatum === datum
                ? "bg-primary-gray text-primary-white"
                : "bg-primary-gray/10 text-primary-gray hover:bg-primary-gray/20"
            }`}
          >
            {datum.replace("-", " ")}
          </button>
        ))}
      </div>

      {/* Chart */}
      <div className="w-full" ref={chartRef} />

      {/* Privacy Toggle */}
      <div className="flex gap-2 md:gap-4 items-center">
        <button
          onClick={() => setIsPrivate(true)}
          className={`px-3 md:px-4 py-2 rounded-lg transition-colors text-sm md:text-base ${
            isPrivate
              ? "bg-accent text-primary-white"
              : "bg-primary-gray/10 text-primary-gray hover:bg-primary-gray/20"
          }`}
        >
          Private
        </button>
        <button
          onClick={() => setIsPrivate(false)}
          className={`px-3 md:px-4 py-2 rounded-lg transition-colors text-sm md:text-base ${
            !isPrivate
              ? "bg-primary-black text-primary-white"
              : "bg-primary-gray/10 text-primary-gray hover:bg-primary-gray/20"
          }`}
        >
          Non-Private
        </button>
      </div>
    </div>
  );
}
