import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronLeftIcon, ChevronRightIcon } from "@heroicons/react/24/outline";
import { getPublicPath } from "@/lib/utils";
import papersData from "@/data/papers.json";
import Image from "next/image";

interface contentBlock {
  type: "text" | "image";
  content?: string;
  src?: string;
  alt?: string;
  caption?: string;
}

interface Paper {
  id: number;
  shortTitle: string;
  title: string;
  author: string;
  thumbnail: string;
  analysis: contentBlock[];
  privatization: contentBlock[];
  results: contentBlock[];
}

export function PaperDisplay() {
  const [selectedPaper, setSelectedPaper] = useState(0);
  const [direction, setDirection] = useState(1); // 1 for right, -1 for left
  const papers = papersData.papers as Paper[];
  const contentRef = useRef<HTMLDivElement>(null);
  const [imagesLoaded, setImagesLoaded] = useState(false);

  // Preload all thumbnail images
  useEffect(() => {
    const preloadImages = async () => {
      console.log("Preloading images...");
      
      // Preload all paper thumbnails
      const thumbnailPromises = papers.map((paper) => {
        return new Promise<string>((resolve, reject) => {
          const img = new window.Image();
          const src = getPublicPath(paper.thumbnail);
          console.log(`Preloading thumbnail: ${src}`);
          img.src = src;
          img.onload = () => {
            console.log(`Loaded thumbnail: ${src}`);
            resolve(src);
          };
          img.onerror = () => {
            console.error(`Failed to load thumbnail: ${src}`);
            reject(new Error(`Failed to load thumbnail: ${src}`));
          };
        });
      });
      
      // Preload all result images
      const resultImagePromises = papers.flatMap((paper) => 
        paper.results
          .filter(result => result.type === "image" && result.src)
          .map(result => {
            return new Promise<string>((resolve, reject) => {
              const img = new window.Image();
              const src = getPublicPath(result.src || "");
              console.log(`Preloading result image: ${src}`);
              img.src = src;
              img.onload = () => {
                console.log(`Loaded result image: ${src}`);
                resolve(src);
              };
              img.onerror = () => {
                console.error(`Failed to load result image: ${src}`);
                reject(new Error(`Failed to load result image: ${src}`));
              };
            });
          })
      );
      
      try {
        await Promise.all([...thumbnailPromises, ...resultImagePromises]);
        console.log("All images preloaded successfully");
        setImagesLoaded(true);
      } catch (error) {
        console.error("Error preloading images:", error);
        // Continue even if some images failed to load
        setImagesLoaded(true);
      }
    };
    
    preloadImages();
  }, [papers]);

  const scrollToTop = () => {
    // Add a small delay to ensure state updates have completed
    setTimeout(() => {
      if (contentRef.current) {
        console.log("Scrolling to top", contentRef.current);
        
        // Try scrollIntoView first
        contentRef.current.scrollIntoView({ 
          behavior: "smooth",
          block: "start"
        });
        
        // Fallback: also try to find the element by ID and scroll to it
        const contentElement = document.getElementById('paper-content');
        if (contentElement) {
          // Scroll the parent container
          const container = document.querySelector('.h-screen.overflow-y-auto');
          if (container) {
            const offsetTop = contentElement.offsetTop;
            container.scrollTo({
              top: offsetTop - 100, // Adjust for header/padding
              behavior: 'smooth'
            });
          }
        }
      } else {
        console.error("contentRef is not attached to any element");
      }
    }, 50);
  };

  const nextPaper = () => {
    setDirection(1);
    setSelectedPaper((prev) => {
      const newIndex = (prev + 1) % papers.length;
      // Use setTimeout to ensure state is updated before scrolling
      setTimeout(() => scrollToTop(), 10);
      return newIndex;
    });
  };

  const previousPaper = () => {
    setDirection(-1);
    setSelectedPaper((prev) => {
      const newIndex = (prev - 1 + papers.length) % papers.length;
      // Use setTimeout to ensure state is updated before scrolling
      setTimeout(() => scrollToTop(), 10);
      return newIndex;
    });
  };

  const selectPaper = (index: number) => {
    setDirection(index > selectedPaper ? 1 : -1);
    setSelectedPaper(index);
    // Use setTimeout to ensure state is updated before scrolling
    setTimeout(() => scrollToTop(), 10);
  };

  return (
    <div className="flex min-h-screen mb-16">
      {/* Left side - Static content */}
      <div className="w-1/2 bg-accent-light p-12 pl-24 sticky top-0 h-screen flex flex-col">
        {/* Tabs at the top */}
        <div className="flex justify-center w-full pt-8">
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

        {/* Paper Stack Container */}
        <div className="flex-1 flex flex-col items-center justify-center">
          {/* Paper Stack with max height constraint */}
          <div className="relative w-full max-w-[350px] max-h-[80%] aspect-[8.5/11] mx-auto">
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
                  {imagesLoaded ? (
                    <Image
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
                      width={510}
                      height={660}
                      priority
                    />
                  ) : (
                    <div className="animate-pulse flex flex-col items-center justify-center">
                      <div className="w-32 h-32 bg-primary-gray/20 rounded-full mb-4"></div>
                      <div className="h-2 w-24 bg-primary-gray/20 rounded mb-2"></div>
                      <div className="h-2 w-16 bg-primary-gray/20 rounded"></div>
                    </div>
                  )}
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
                    },
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
                    },
                  },
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
                  },
                }}
                className="absolute inset-0"
                style={{
                  perspective: "1000px",
                  transformStyle: "preserve-3d",
                }}
              >
                <div className="w-full h-full bg-primary-white rounded-lg shadow-xl border border-primary-gray/10">
                  <div className="w-full h-full flex items-center justify-center text-primary-gray">
                    {imagesLoaded ? (
                      <Image
                        src={getPublicPath(papers[selectedPaper].thumbnail)}
                        alt={`${papers[selectedPaper].shortTitle} Thumbnail`}
                        className="w-full h-full object-contain"
                        width={510}
                        height={660}
                        priority
                      />
                    ) : (
                      <div className="animate-pulse flex flex-col items-center justify-center">
                        <div className="w-32 h-32 bg-primary-gray/20 rounded-full mb-4"></div>
                        <div className="h-2 w-24 bg-primary-gray/20 rounded mb-2"></div>
                        <div className="h-2 w-16 bg-primary-gray/20 rounded"></div>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Navigation arrows at the bottom */}
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
      <div className="w-3/4 p-12 pt-28 mt-16" ref={contentRef} id="paper-content">
        <div className="max-w-3xl mx-auto space-y-16">
          {/* Paper Title */}
          <section>
            <h1 className="text-4xl font-bold mb-4 text-primary-gray">
              {papers[selectedPaper].id}. {papers[selectedPaper].title}
            </h1>
            <p className="text-xl text-primary-gray">
              Primary contributor: {papers[selectedPaper].author}
            </p>
          </section>

          {/* Paper Analysis */}
          <section>
            <h2 className="text-3xl font-bold mb-4 text-primary-gray">
              {papers[selectedPaper].id}.1 Paper&apos;s Analysis
            </h2>
            <div className="space-y-8">
              {papers[selectedPaper].analysis.map((block, index) => {
                if (block.type === "text") {
                  return (
                    <p key={index} className="text-xl text-primary-gray">
                      {block.content}
                    </p>
                  );
                } else if (block.type === "image") {
                  return (
                    <figure key={index} className="my-8">
                      {imagesLoaded ? (
                        <Image
                          src={getPublicPath(block.src || "")}
                          alt={block.alt || ""}
                          width={800}
                          height={400}
                          className="w-2/3 h-auto mx-auto"
                          sizes="(max-width: 768px) 100vw, 800px"
                          priority
                        />
                      ) : (
                        <div className="w-2/3 h-64 mx-auto bg-primary-gray/10 animate-pulse flex items-center justify-center">
                          <div className="text-primary-gray/50">Loading image...</div>
                        </div>
                      )}
                      <figcaption className="text-center text-sm mt-2 text-primary-gray">
                        {block.caption}
                      </figcaption>
                    </figure>
                  );
                }
              })}
            </div>
          </section>

          {/* Privatization Approach */}
          <section>
            <h2 className="text-3xl font-bold mb-4 text-primary-gray">
              {papers[selectedPaper].id}.2 How We Privatized
            </h2>
            <div className="space-y-8">
              {papers[selectedPaper].privatization.map((block, index) => {
                if (block.type === "text") {
                  return (
                    <p key={index} className="text-xl text-primary-gray">
                      {block.content}
                    </p>
                  );
                } else if (block.type === "image") {
                  return (
                    <figure key={index} className="my-8">
                      {imagesLoaded ? (
                        <Image
                          src={getPublicPath(block.src || "")}
                          alt={block.alt || ""}
                          width={800}
                          height={400}
                          className="w-2/3 h-auto mx-auto"
                          sizes="(max-width: 768px) 100vw, 800px"
                          priority
                        />
                      ) : (
                        <div className="w-2/3 h-64 mx-auto bg-primary-gray/10 animate-pulse flex items-center justify-center">
                          <div className="text-primary-gray/50">Loading image...</div>
                        </div>
                      )}
                      <figcaption className="text-center text-sm mt-2 text-primary-gray">
                        {block.caption}
                      </figcaption>
                    </figure>
                  );
                }
              })}
            </div>
          </section>

          {/* Results */}
          <section>
            <h2 className="text-3xl font-bold mb-4 text-primary-gray">
              {papers[selectedPaper].id}.3 Results Compared to Paper
            </h2>
            <div className="space-y-8">
              {papers[selectedPaper].results.map((block, index) => {
                if (block.type === "text") {
                  return (
                    <p key={index} className="text-xl text-primary-gray">
                      {block.content}
                    </p>
                  );
                } else if (block.type === "image") {
                  return (
                    <figure key={index} className="my-8">
                      {imagesLoaded ? (
                        <Image
                          src={getPublicPath(block.src || "")}
                          alt={block.alt || ""}
                          width={800}
                          height={400}
                          className="w-2/3 h-auto mx-auto"
                          sizes="(max-width: 768px) 100vw, 800px"
                          priority
                        />
                      ) : (
                        <div className="w-2/3 h-64 mx-auto bg-primary-gray/10 animate-pulse flex items-center justify-center">
                          <div className="text-primary-gray/50">Loading image...</div>
                        </div>
                      )}
                      <figcaption className="text-center text-sm mt-2 text-primary-gray">
                        {block.caption}
                      </figcaption>
                    </figure>
                  );
                }
              })}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
