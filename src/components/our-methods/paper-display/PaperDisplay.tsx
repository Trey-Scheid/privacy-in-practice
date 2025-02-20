import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronLeftIcon, ChevronRightIcon } from "@heroicons/react/24/outline";

interface Paper {
  id: number;
  title: string;
  thumbnail: string;
  analysis: string;
  privatization: string;
  results: string;
}

// Sample data - replace with actual content
const papers: Paper[] = [
  {
    id: 1,
    title: "Paper One",
    thumbnail: "/paper-placeholder.png",
    analysis: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    privatization: "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
    results: "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.",
  },
  {
    id: 2,
    title: "Paper Two",
    thumbnail: "/paper-placeholder.png",
    analysis: "Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt.",
    privatization: "Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem.",
    results: "Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur.",
  },
  {
    id: 3,
    title: "Paper Three",
    thumbnail: "/paper-placeholder.png",
    analysis: "At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident.",
    privatization: "Similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga. Et harum quidem rerum facilis est et expedita distinctio.",
    results: "Nam libero tempore, cum soluta nobis est eligendi optio cumque nihil impedit quo minus id quod maxime placeat facere possimus, omnis voluptas assumenda est, omnis dolor repellendus.",
  },
  {
    id: 4,
    title: "Paper Four",
    thumbnail: "/paper-placeholder.png",
    analysis: "Temporibus autem quibusdam et aut officiis debitis aut rerum necessitatibus saepe eveniet ut et voluptates repudiandae sint et molestiae non recusandae.",
    privatization: "Itaque earum rerum hic tenetur a sapiente delectus, ut aut reiciendis voluptatibus maiores alias consequatur aut perferendis doloribus asperiores repellat.",
    results: "On the other hand, we denounce with righteous indignation and dislike men who are so beguiled and demoralized by the charms of pleasure of the moment.",
  },
];

export function PaperDisplay() {
  const [selectedPaper, setSelectedPaper] = useState(0);
  const [direction, setDirection] = useState(1); // 1 for right, -1 for left

  const nextPaper = () => {
    setDirection(1);
    setSelectedPaper((prev) => (prev + 1) % papers.length);
  };

  const previousPaper = () => {
    setDirection(-1);
    setSelectedPaper((prev) => (prev - 1 + papers.length) % papers.length);
  };

  const selectPaper = (index: number) => {
    setDirection(index > selectedPaper ? 1 : -1);
    setSelectedPaper(index);
  };

  return (
    <div className="flex min-h-screen">
      {/* Left side - Static content */}
      <div className="w-1/3 bg-primary-white p-8 sticky top-0 h-screen flex flex-col">
        {/* Tabs */}
        <div className="flex flex-col gap-2 mb-8">
          {papers.map((paper, index) => (
            <button
              key={paper.id}
              onClick={() => selectPaper(index)}
              className={`p-4 text-left rounded-lg transition-colors ${
                selectedPaper === index
                  ? "bg-accent text-primary-white"
                  : "bg-primary-gray/10 text-primary-gray hover:bg-primary-gray/20"
              }`}
            >
              {paper.title}
            </button>
          ))}
        </div>

        {/* Paper Stack */}
        <div className="flex-1 relative">
          <div className="relative w-full aspect-[8.5/11] mx-auto">
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
                  {papers[(selectedPaper - 1 + papers.length) % papers.length].title} Thumbnail
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
                animate={{
                  x: 0,
                  y: 0,
                  rotate: 0,
                  scale: 1,
                  zIndex: 20,
                }}
                exit={{
                  x: direction > 0 ? -100 : 100,
                  y: direction > 0 ? 100 : -100,
                  rotate: direction > 0 ? -10 : 10,
                  scale: 0.9,
                  zIndex: 0,
                }}
                transition={{
                  type: "spring",
                  stiffness: 300,
                  damping: 30,
                }}
                className="absolute inset-0"
                style={{
                  perspective: '1000px',
                  transformStyle: 'preserve-3d',
                }}
              >
                <div className="w-full h-full bg-primary-white rounded-lg shadow-xl border border-primary-gray/10">
                  <div className="w-full h-full flex items-center justify-center text-primary-gray">
                    {papers[selectedPaper].title} Thumbnail
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Navigation arrows */}
          <div className="flex justify-center gap-4 mt-8">
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
      <div className="w-2/3 p-8">
        <div className="max-w-3xl mx-auto space-y-16">
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
