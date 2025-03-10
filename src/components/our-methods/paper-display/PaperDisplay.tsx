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
  algorithm: string;
  thumbnail: string;
  analysis: contentBlock[];
  privatization: contentBlock[];
  results: contentBlock[];
  interpretation: contentBlock[];
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
          .filter((result) => result.type === "image" && result.src)
          .map((result) => {
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
    // Only scroll on desktop screens (lg breakpoint and above)
    if (window.innerWidth >= 1024) {
      // Add a small delay to ensure state updates have completed
      setTimeout(() => {
        if (contentRef.current) {
          console.log("Scrolling to top", contentRef.current);

          // Try scrollIntoView first
          contentRef.current.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });

          // Fallback: also try to find the element by ID and scroll to it
          const contentElement = document.getElementById("paper-content");
          if (contentElement) {
            // Scroll the parent container
            const container = document.querySelector(
              ".h-screen.overflow-y-auto"
            );
            if (container) {
              const offsetTop = contentElement.offsetTop;
              container.scrollTo({
                top: offsetTop - 100, // Adjust for header/padding
                behavior: "smooth",
              });
            }
          }
        } else {
          console.error("contentRef is not attached to any element");
        }
      }, 50);
    }
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
    <div className="flex flex-col lg:flex-row min-h-screen mb-16">
      {/* Mobile layout - Paper selection at the top */}
      <div className="lg:hidden w-full bg-accent-light p-4">
        {/* Tabs at the top */}
        <div className="flex flex-wrap justify-center w-full py-4">
          {papers.map((paper, index) => (
            <button
              key={paper.id}
              onClick={() => selectPaper(index)}
              className={`px-3 py-2 border-b-2 transition-colors text-sm ${
                selectedPaper === index
                  ? "border-accent text-accent font-medium"
                  : "border-transparent text-primary-gray hover:border-primary-gray/20"
              }`}
            >
              {paper.shortTitle}
            </button>
          ))}
        </div>

        {/* Current paper thumbnail */}
        <div className="flex justify-center mt-2 mb-6">
          <div className="relative w-40 aspect-[8.5/11]">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={selectedPaper}
                initial={{
                  opacity: 0,
                  x: direction > 0 ? 20 : -20,
                }}
                animate={{
                  opacity: 1,
                  x: 0,
                  transition: { duration: 0.3 },
                }}
                exit={{
                  opacity: 0,
                  x: direction > 0 ? -20 : 20,
                  transition: { duration: 0.3 },
                }}
                className="absolute inset-0"
              >
                <div className="w-full h-full bg-primary-white rounded-lg shadow-md border border-primary-gray/10">
                  <div className="w-full h-full flex items-center justify-center">
                    {imagesLoaded ? (
                      <Image
                        src={getPublicPath(papers[selectedPaper].thumbnail)}
                        alt={`${papers[selectedPaper].shortTitle} Thumbnail`}
                        className="w-full h-full object-contain"
                        width={200}
                        height={260}
                        priority
                      />
                    ) : (
                      <div className="animate-pulse flex flex-col items-center justify-center">
                        <div className="w-16 h-16 bg-primary-gray/20 rounded-full mb-2"></div>
                        <div className="h-2 w-12 bg-primary-gray/20 rounded mb-1"></div>
                        <div className="h-2 w-8 bg-primary-gray/20 rounded"></div>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>
        </div>

        {/* Navigation arrows */}
        <div className="flex justify-center gap-4 mb-2">
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

      {/* Desktop layout - Static sidebar */}
      <div className="hidden lg:block lg:w-1/3 bg-accent-light p-12 pl-24 sticky top-0 h-screen flex flex-col">
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
                      width={300}
                      height={400}
                      priority
                    />
                  ) : (
                    <div className="animate-pulse flex flex-col items-center justify-center">
                      <div className="w-24 h-24 bg-primary-gray/20 rounded-full mb-4"></div>
                      <div className="h-2 w-16 bg-primary-gray/20 rounded mb-2"></div>
                      <div className="h-2 w-12 bg-primary-gray/20 rounded"></div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Main paper with animation */}
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={selectedPaper}
                initial={{
                  opacity: 0,
                  x: direction > 0 ? 20 : -20,
                }}
                animate={{
                  opacity: 1,
                  x: 0,
                  transition: { duration: 0.3 },
                }}
                exit={{
                  opacity: 0,
                  x: direction > 0 ? -20 : 20,
                  transition: { duration: 0.3 },
                }}
                className="absolute inset-0"
              >
                <div className="w-full h-full bg-primary-white rounded-lg shadow-xl border border-primary-gray/10">
                  <div className="w-full h-full flex items-center justify-center text-primary-gray">
                    {imagesLoaded ? (
                      <Image
                        src={getPublicPath(papers[selectedPaper].thumbnail)}
                        alt={`${papers[selectedPaper].shortTitle} Thumbnail`}
                        className="w-full h-full object-contain"
                        width={300}
                        height={400}
                        priority
                      />
                    ) : (
                      <div className="animate-pulse flex flex-col items-center justify-center">
                        <div className="w-24 h-24 bg-primary-gray/20 rounded-full mb-4"></div>
                        <div className="h-2 w-16 bg-primary-gray/20 rounded mb-2"></div>
                        <div className="h-2 w-12 bg-primary-gray/20 rounded"></div>
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

      {/* Content area */}
      <div
        className="w-full lg:w-3/4 p-4 md:p-12 pt-6 lg:pt-28"
        ref={contentRef}
        id="paper-content"
      >
        <div className="max-w-3xl mx-auto space-y-8 md:space-y-16">
          {/* Paper Title */}
          <section>
            <h1 className="text-2xl md:text-4xl font-bold mb-2 md:mb-4 text-primary-gray">
              {papers[selectedPaper].id}. {papers[selectedPaper].title}
            </h1>
            <p className="text-lg md:text-xl text-primary-gray">
              Primary contributor: {papers[selectedPaper].author}
            </p>
            <p className="text-lg md:text-xl text-primary-gray">
              Algorithm used: {papers[selectedPaper].algorithm}
            </p>
          </section>

          {/* Paper Analysis */}
          <section>
            <h2 className="text-xl md:text-3xl font-bold mb-2 md:mb-4 text-primary-gray">
              {papers[selectedPaper].id}.1 Paper&apos;s Analysis
            </h2>
            <div className="space-y-4 md:space-y-8">
              {papers[selectedPaper].analysis.map((block, index) => {
                if (block.type === "text") {
                  return (
                    <p
                      key={index}
                      className="text-base md:text-xl text-primary-gray"
                    >
                      {block.content}
                    </p>
                  );
                } else if (block.type === "image") {
                  return (
                    <figure key={index} className="my-4 md:my-8">
                      {imagesLoaded ? (
                        <div className="flex flex-col items-center">
                          <Image
                            src={getPublicPath(block.src || "")}
                            alt={block.alt || ""}
                            width={800}
                            height={600}
                            className="rounded-lg max-w-full w-full md:w-2/3 h-auto"
                            priority
                          />
                          <figcaption className="mt-2 text-sm md:text-base text-center text-primary-gray">
                            {block.caption}
                          </figcaption>
                        </div>
                      ) : (
                        <div className="w-full aspect-video bg-primary-gray/10 animate-pulse rounded-lg flex items-center justify-center">
                          <div className="text-primary-gray/40">
                            Loading image...
                          </div>
                        </div>
                      )}
                    </figure>
                  );
                }
                return null;
              })}
            </div>
          </section>

          {/* Paper Privatization */}
          <section>
            <h2 className="text-xl md:text-3xl font-bold mb-2 md:mb-4 text-primary-gray">
              {papers[selectedPaper].id}.2 Privatization
            </h2>
            <div className="space-y-4 md:space-y-8">
              {papers[selectedPaper].privatization.map((block, index) => {
                if (block.type === "text") {
                  return (
                    <p
                      key={index}
                      className="text-base md:text-xl text-primary-gray"
                    >
                      {block.content}
                    </p>
                  );
                } else if (block.type === "image") {
                  return (
                    <figure key={index} className="my-4 md:my-8">
                      {imagesLoaded ? (
                        <div className="flex flex-col items-center">
                          <Image
                            src={getPublicPath(block.src || "")}
                            alt={block.alt || ""}
                            width={800}
                            height={600}
                            className="rounded-lg max-w-full w-full md:w-2/3 h-auto"
                            priority
                          />
                          <figcaption className="mt-2 text-sm md:text-base text-center text-primary-gray">
                            {block.caption}
                          </figcaption>
                        </div>
                      ) : (
                        <div className="w-full aspect-video bg-primary-gray/10 animate-pulse rounded-lg flex items-center justify-center">
                          <div className="text-primary-gray/40">
                            Loading image...
                          </div>
                        </div>
                      )}
                    </figure>
                  );
                }
                return null;
              })}
            </div>
          </section>

          {/* Paper Results */}
          <section>
            <h2 className="text-xl md:text-3xl font-bold mb-2 md:mb-4 text-primary-gray">
              {papers[selectedPaper].id}.3 Results
            </h2>
            <div className="space-y-4 md:space-y-8">
              {papers[selectedPaper].results.map((block, index) => {
                if (block.type === "text") {
                  return (
                    <p
                      key={index}
                      className="text-base md:text-xl text-primary-gray"
                    >
                      {block.content}
                    </p>
                  );
                } else if (block.type === "image") {
                  return (
                    <figure key={index} className="my-4 md:my-8">
                      {imagesLoaded ? (
                        <div className="flex flex-col items-center">
                          <Image
                            src={getPublicPath(block.src || "")}
                            alt={block.alt || ""}
                            width={800}
                            height={600}
                            className="rounded-lg max-w-full w-full md:w-2/3 h-auto"
                            priority
                          />
                          <figcaption className="mt-2 text-sm md:text-base text-center text-primary-gray">
                            {block.caption}
                          </figcaption>
                        </div>
                      ) : (
                        <div className="w-full aspect-video bg-primary-gray/10 animate-pulse rounded-lg flex items-center justify-center">
                          <div className="text-primary-gray/40">
                            Loading image...
                          </div>
                        </div>
                      )}
                    </figure>
                  );
                }
                return null;
              })}
            </div>
          </section>
          {/* Paper Interpretation */}
          <section>
            <h2 className="text-xl md:text-3xl font-bold mb-2 md:mb-4 text-primary-gray">
              {papers[selectedPaper].id}.4 Interpretation
            </h2>
            <div className="space-y-4 md:space-y-8">
              {papers[selectedPaper].interpretation.map((block, index) => {
                if (block.type === "text") {
                  return (
                    <p
                      key={index}
                      className="text-base md:text-xl text-primary-gray"
                    >
                      {block.content}
                    </p>
                  );
                } else if (block.type === "image") {
                  return (
                    <figure key={index} className="my-4 md:my-8">
                      {imagesLoaded ? (
                        <div className="flex flex-col items-center">
                          <Image
                            src={getPublicPath(block.src || "")}
                            alt={block.alt || ""}
                            width={800}
                            height={600}
                            className="rounded-lg max-w-full w-full md:w-2/3 h-auto"
                            priority
                          />
                          <figcaption className="mt-2 text-sm md:text-base text-center text-primary-gray">
                            {block.caption}
                          </figcaption>
                        </div>
                      ) : (
                        <div className="w-full aspect-video bg-primary-gray/10 animate-pulse rounded-lg flex items-center justify-center">
                          <div className="text-primary-gray/40">
                            Loading image...
                          </div>
                        </div>
                      )}
                    </figure>
                  );
                }
                return null;
              })}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
