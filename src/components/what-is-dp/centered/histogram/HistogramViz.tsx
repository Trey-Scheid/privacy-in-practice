import { useState, useEffect } from "react";
import { Chart } from "./Chart";

interface DataRow {
  raw: number;
  noise: number;
}

export function HistogramViz() {
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

  // Update data when scale or epsilon changes
  const updateData = () => {
    setData(
      baseData.map((row) => ({
        raw: row.raw * scale,
        noise: generateLaplaceNoise(1 / (epsilon / 3)),
      }))
    );
  };

  // Update data when scale changes
  useEffect(() => {
    setData((currentData) =>
      baseData.map((baseRow, index) => ({
        raw: baseRow.raw * scale,
        noise: currentData[index].noise,
      }))
    );
  }, [scale]);

  return (
    <>
      <div className="prose prose-lg max-w-3xl mx-auto mb-8 md:mb-12 text-center">
        <p className="text-base md:text-xl">
          Another simple way to privatize data is to add noise to each of the
          values you plan to release. Here, we&apos;re releasing the raw counts
          of each category by adding noise to each of the counts.
        </p>
      </div>
      <div className="flex flex-col md:flex-row justify-between items-start gap-8">
        {/* Table (full width on mobile, half width on desktop) */}
        <div className="w-full md:w-1/2">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="p-3 md:p-4 text-left border-b border-primary-gray/10">
                  <div className="flex items-center gap-2">
                    <div>Raw Data</div>
                    <div className="relative group">
                      <div className="w-5 h-5 rounded-full bg-primary-gray/10 flex items-center justify-center text-primary-gray cursor-help">
                        ?
                      </div>
                      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-64 p-3 bg-primary-gray text-primary-white rounded-lg text-sm opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-10">
                        Notice how having more data means that the noise affects
                        the big picture less.
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
                      onChange={(e) => {
                        setScale(scaleValues[parseInt(e.target.value)]);
                      }}
                      className="w-full h-2 bg-primary-gray rounded-lg appearance-none cursor-pointer accent-accent"
                    />
                    <div className="text-sm text-primary-gray mt-1">
                      Scale ={" "}
                      {scale < 1 ? `1/${Math.round(1 / scale)}` : `${scale}`}x
                    </div>
                  </div>
                </th>
                <th className="p-3 md:p-4 text-left border-b border-primary-gray/10">
                  <div className="flex items-center gap-2">
                    <div>Introduce Noise</div>
                    <div className="relative group">
                      <div className="w-5 h-5 rounded-full bg-primary-gray/10 flex items-center justify-center text-primary-gray cursor-help">
                        ?
                      </div>
                      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-64 p-3 bg-primary-gray text-primary-white rounded-lg text-sm opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-10">
                        The noise follows a Laplace distribution with scale
                        proportional to 1/ε. This mechanism satisfies (ε,
                        0)-differential privacy, meaning that the probability of
                        the privacy guarantee failing is 0.
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
                      onChange={(e) => {
                        setEpsilon(parseFloat(e.target.value));
                        updateData();
                      }}
                      className="w-full h-2 bg-primary-gray rounded-lg appearance-none cursor-pointer accent-accent"
                    />
                    <div className="text-sm text-primary-gray mt-1">
                      ε = {epsilon.toFixed(1)}
                    </div>
                  </div>
                </th>
                <th className="p-3 md:p-4 text-left border-b border-primary-gray/10">
                  Privatized Data
                </th>
              </tr>
            </thead>
            <tbody>
              {data.map((row, i) => (
                <tr key={i}>
                  <td className="p-3 md:p-4 border-b border-primary-gray/10">
                    {Math.round(row.raw * 100) / 100}
                  </td>
                  <td className="p-3 md:p-4 border-b border-primary-gray/10">
                    {row.noise.toFixed(2)}
                  </td>
                  <td className="p-3 md:p-4 border-b border-primary-gray/10">
                    {(row.raw + row.noise).toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {/* Chart (full width on mobile, half width on desktop) */}
        <div className="w-full md:w-1/2 mt-8 md:mt-0">
          <Chart data={data} />
        </div>
      </div>
    </>
  );
}
