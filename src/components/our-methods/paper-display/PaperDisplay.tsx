import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronLeftIcon, ChevronRightIcon } from "@heroicons/react/24/outline";
import { getPublicPath } from "@/lib/utils";
import papersData from "@/data/papers.json";

export function PaperDisplay() {
  const [selectedPaper, setSelectedPaper] = useState(0);
  const [direction, setDirection] = useState(1); // 1 for right, -1 for left
  const papers = papersData.papers;
  const contentRef = useRef<HTMLDivElement>(null);

  const scrollToTop = () => {
    contentRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const nextPaper = () => {
    setDirection(1);
    setSelectedPaper((prev) => (prev + 1) % papers.length);
    scrollToTop();
  };

  const previousPaper = () => {
    setDirection(-1);
    setSelectedPaper((prev) => (prev - 1 + papers.length) % papers.length);
    scrollToTop();
  };

  const selectPaper = (index: number) => {
    setDirection(index > selectedPaper ? 1 : -1);
    setSelectedPaper(index);
    scrollToTop();
  };

  console.log(papers[0].thumbnail);

  return (
    <div className="flex min-h-screen mb-16">
      {/* Left side - Static content */}
      <div className="w-1/2 bg-accent-light p-12 pl-24 sticky top-0 h-screen flex flex-col">
        {/* Paper Stack with Tabs */}
        <div className="flex-1 relative flex flex-col justify-center h-full">
          {/* Tabs positioned above stack */}
          <div className="flex w-full absolute top-24 z-30 bg-accent-light">
            {papers.map((paper, index) => (
              <button
                key={paper.id}
                onClick={() => selectPaper(index)}
                className={`px-4 py-2 border-b-2 transition-colors ${
                  selectedPaper === index
                    ? "border-accent text-accent font-medium"
                    : "border-transparent text-primary-gray hover:border-primary-gray/20"
                }`}
              >
                {paper.shortTitle}
              </button>
            ))}
          </div>

          {/* Paper Stack */}
          <div className="relative w-full aspect-[8.5/11] mx-auto mt-32 flex-2">
            {/* Static background papers */}
            {[3, 2, 1].map((offset) => (
              <div
                key={`stack-${offset}`}
                className="absolute inset-0"
                style={{
                  transform: `translate(${offset * 4}px, ${offset * 4}px)`,
                }}
              >
                <div className="w-full h-full bg-primary-gray/10 rounded-lg shadow-lg" />
              </div>
            ))}

            {/* Previous paper stays in stack */}
            <div
              className="absolute inset-0"
              style={{
                transform: `translate(12px, 12px)`,
              }}
            >
              <div className="w-full h-full bg-primary-white rounded-lg shadow-xl border border-primary-gray/10">
                <div className="w-full h-full flex items-center justify-center text-primary-gray">
                  <img
                    src={getPublicPath(
                      papers[
                        (selectedPaper - 1 + papers.length) % papers.length
                      ].thumbnail
                    )}
                    alt={`${
                      papers[
                        (selectedPaper - 1 + papers.length) % papers.length
                      ].shortTitle
                    } Thumbnail`}
                    className="w-full h-full object-contain"
                  />
                </div>
              </div>
            </div>

            {/* Animated current paper */}
            <AnimatePresence mode="popLayout">
              <motion.div
                key={selectedPaper}
                initial={{
                  x: 12,
                  y: 12,
                  rotate: 0,
                  scale: 1,
                  zIndex: 10,
                }}
                variants={{
                  pull: {
                    x: direction > 0 ? 24 : -24,
                    y: direction > 0 ? -24 : 24,
                    rotate: direction > 0 ? 5 : -5,
                    scale: 1.02,
                    zIndex: 20,
                    transition: {
                      duration: 0.2,
                    }
                  },
                  settle: {
                    x: 0,
                    y: 0,
                    rotate: 0,
                    scale: 1,
                    zIndex: 20,
                    transition: {
                      type: "spring",
                      stiffness: 300,
                      damping: 25,
                    }
                  }
                }}
                animate={["pull", "settle"]}
                exit={{
                  x: 12,
                  y: 12,
                  rotate: 0,
                  scale: 0.98,
                  zIndex: 0,
                  transition: {
                    duration: 0.2,
                  }
                }}
                className="absolute inset-0"
                style={{
                  perspective: "1000px",
                  transformStyle: "preserve-3d",
                }}
              >
                <div className="w-full h-full bg-primary-white rounded-lg shadow-xl border border-primary-gray/10">
                  <div className="w-full h-full flex items-center justify-center text-primary-gray">
                    <img
                      src={getPublicPath(papers[selectedPaper].thumbnail)}
                      alt={`${papers[selectedPaper].shortTitle} Thumbnail`}
                      className="w-full h-full object-contain"
                    />
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Navigation arrows */}
          <div className="flex justify-center gap-4 mt-12">
            <button
              onClick={previousPaper}
              className="p-2 rounded-full bg-primary-gray/10 hover:bg-primary-gray/20 transition-colors"
            >
              <ChevronLeftIcon className="w-6 h-6 text-primary-gray" />
            </button>
            <button
              onClick={nextPaper}
              className="p-2 rounded-full bg-primary-gray/10 hover:bg-primary-gray/20 transition-colors"
            >
              <ChevronRightIcon className="w-6 h-6 text-primary-gray" />
            </button>
          </div>
        </div>
      </div>

      {/* Right side - Scrollable content */}
      <div className="w-3/4 p-12 pt-28 mt-16" ref={contentRef}>
        <div className="max-w-3xl mx-auto space-y-16">
          {/* Paper Title */}
          <section>
            <h1 className="text-4xl font-bold mb-4 text-primary-gray">
              {papers[selectedPaper].title}
            </h1>
          </section>

          {/* Paper Analysis */}
          <section>
            <h2 className="text-3xl font-bold mb-4 text-primary-gray">
              Paper&apos;s Analysis
            </h2>
            <p className="text-xl text-primary-gray">
              {papers[selectedPaper].analysis}
            </p>
          </section>

          {/* Privatization Approach */}
          <section>
            <h2 className="text-3xl font-bold mb-4 text-primary-gray">
              How We Privatized
            </h2>
            <p className="text-xl text-primary-gray">
              {papers[selectedPaper].privatization}
            </p>
          </section>

          {/* Results */}
          <section>
            <h2 className="text-3xl font-bold mb-4 text-primary-gray">
              Results Compared to Paper
            </h2>
            <p className="text-xl text-primary-gray">
              {papers[selectedPaper].results}
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
